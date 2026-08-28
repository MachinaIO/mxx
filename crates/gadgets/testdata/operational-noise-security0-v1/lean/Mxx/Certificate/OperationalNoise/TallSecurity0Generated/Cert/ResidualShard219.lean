import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard218

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult28617
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 28617
end ResidualResult28617

namespace ResidualResult28621
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28617.actual selector witness -
    ResidualResult28614.actual selector witness
end ResidualResult28621

namespace ResidualResult28625
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28621.actual selector witness -
    ResidualResult28606.actual selector witness
end ResidualResult28625

namespace ResidualResult28634
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult28463.actual selector witness
end ResidualResult28634

namespace ResidualResult28641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28634.actual selector witness +
    ResidualResult28456.actual selector witness
end ResidualResult28641

namespace ResidualResult28648
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 28648
end ResidualResult28648

namespace ResidualResult28651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 28651
end ResidualResult28651

namespace ResidualResult28658
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 28658
end ResidualResult28658

namespace ResidualResult28661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 28661
end ResidualResult28661

namespace ResidualResult28666
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1187.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult28666

namespace ResidualResult28671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult13987.actual selector witness
end ResidualResult28671

namespace ResidualResult28675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28671.actual selector witness -
    ResidualResult28666.actual selector witness
end ResidualResult28675

namespace ResidualResult28681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28675.actual selector witness +
    ResidualResult13979.actual selector witness
end ResidualResult28681

namespace ResidualResult28689
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult28681.actual selector witness *
    ResidualResult1190.actual selector witness
end ResidualResult28689

namespace ResidualResult28694
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1190.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult28694

namespace ResidualResult28699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult14028.actual selector witness
end ResidualResult28699

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
