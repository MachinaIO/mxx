import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult69641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 69641
end ResidualResult69641

namespace ResidualResult69644
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 69644
end ResidualResult69644

namespace ResidualResult69649
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3293.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult69649

namespace ResidualResult69654
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult10981.actual selector witness
end ResidualResult69654

namespace ResidualResult69658
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69654.actual selector witness -
    ResidualResult69649.actual selector witness
end ResidualResult69658

namespace ResidualResult69664
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69658.actual selector witness +
    ResidualResult10973.actual selector witness
end ResidualResult69664

namespace ResidualResult69672
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69664.actual selector witness *
    ResidualResult3296.actual selector witness
end ResidualResult69672

namespace ResidualResult69677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3296.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult69677

namespace ResidualResult69682
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult11022.actual selector witness
end ResidualResult69682

namespace ResidualResult69686
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69682.actual selector witness -
    ResidualResult69677.actual selector witness
end ResidualResult69686

namespace ResidualResult69692
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69686.actual selector witness +
    ResidualResult11014.actual selector witness
end ResidualResult69692

namespace ResidualResult69702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69692.actual selector witness *
    ResidualResult11011.actual selector witness
end ResidualResult69702

namespace ResidualResult69708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69702.actual selector witness +
    ResidualResult69672.actual selector witness
end ResidualResult69708

namespace ResidualResult69718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult69708.actual selector witness *
    ResidualResult69644.actual selector witness
end ResidualResult69718

namespace ResidualResult69721
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 69721
end ResidualResult69721

namespace ResidualResult69725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 69725
end ResidualResult69725

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
