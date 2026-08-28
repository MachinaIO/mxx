import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard491

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult68708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68700.actual selector witness *
    ResidualResult3250.actual selector witness
end ResidualResult68708

namespace ResidualResult68713
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3250.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult68713

namespace ResidualResult68718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult10020.actual selector witness
end ResidualResult68718

namespace ResidualResult68722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68718.actual selector witness -
    ResidualResult68713.actual selector witness
end ResidualResult68722

namespace ResidualResult68728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68722.actual selector witness +
    ResidualResult10012.actual selector witness
end ResidualResult68728

namespace ResidualResult68738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68728.actual selector witness *
    ResidualResult10009.actual selector witness
end ResidualResult68738

namespace ResidualResult68744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68738.actual selector witness +
    ResidualResult68708.actual selector witness
end ResidualResult68744

namespace ResidualResult68754
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68744.actual selector witness *
    ResidualResult68680.actual selector witness
end ResidualResult68754

namespace ResidualResult68757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68757
end ResidualResult68757

namespace ResidualResult68761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68761
end ResidualResult68761

namespace ResidualResult68839
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68839
end ResidualResult68839

namespace ResidualResult68842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68842
end ResidualResult68842

namespace ResidualResult68847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68842.actual selector witness *
    ResidualResult68839.actual selector witness
end ResidualResult68847

namespace ResidualResult68858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68858
end ResidualResult68858

namespace ResidualResult68861
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68861
end ResidualResult68861

namespace ResidualResult68870
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68870
end ResidualResult68870

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
