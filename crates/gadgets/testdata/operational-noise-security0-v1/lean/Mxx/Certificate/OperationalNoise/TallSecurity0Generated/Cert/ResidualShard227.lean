import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard226

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult29635
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult14989.actual selector witness
end ResidualResult29635

namespace ResidualResult29639
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29635.actual selector witness -
    ResidualResult29630.actual selector witness
end ResidualResult29639

namespace ResidualResult29645
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29639.actual selector witness +
    ResidualResult14981.actual selector witness
end ResidualResult29645

namespace ResidualResult29653
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29645.actual selector witness *
    ResidualResult1236.actual selector witness
end ResidualResult29653

namespace ResidualResult29658
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1236.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult29658

namespace ResidualResult29663
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult15030.actual selector witness
end ResidualResult29663

namespace ResidualResult29667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29663.actual selector witness -
    ResidualResult29658.actual selector witness
end ResidualResult29667

namespace ResidualResult29673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29667.actual selector witness +
    ResidualResult15022.actual selector witness
end ResidualResult29673

namespace ResidualResult29683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29673.actual selector witness *
    ResidualResult15019.actual selector witness
end ResidualResult29683

namespace ResidualResult29689
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29683.actual selector witness +
    ResidualResult29653.actual selector witness
end ResidualResult29689

namespace ResidualResult29699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29689.actual selector witness *
    ResidualResult29625.actual selector witness
end ResidualResult29699

namespace ResidualResult29702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 29702
end ResidualResult29702

namespace ResidualResult29706
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 29706
end ResidualResult29706

namespace ResidualResult29784
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 29784
end ResidualResult29784

namespace ResidualResult29787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 29787
end ResidualResult29787

namespace ResidualResult29792
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult29787.actual selector witness *
    ResidualResult29784.actual selector witness
end ResidualResult29792

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
