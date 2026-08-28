import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult27687
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 27687
end ResidualResult27687

namespace ResidualResult27694
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 27694
end ResidualResult27694

namespace ResidualResult27697
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 27697
end ResidualResult27697

namespace ResidualResult27702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1141.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult27702

namespace ResidualResult27707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult12985.actual selector witness
end ResidualResult27707

namespace ResidualResult27711
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27707.actual selector witness -
    ResidualResult27702.actual selector witness
end ResidualResult27711

namespace ResidualResult27717
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27711.actual selector witness +
    ResidualResult12977.actual selector witness
end ResidualResult27717

namespace ResidualResult27725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27717.actual selector witness *
    ResidualResult1144.actual selector witness
end ResidualResult27725

namespace ResidualResult27730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1144.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult27730

namespace ResidualResult27735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult13026.actual selector witness
end ResidualResult27735

namespace ResidualResult27739
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27735.actual selector witness -
    ResidualResult27730.actual selector witness
end ResidualResult27739

namespace ResidualResult27745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27739.actual selector witness +
    ResidualResult13018.actual selector witness
end ResidualResult27745

namespace ResidualResult27755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27745.actual selector witness *
    ResidualResult13015.actual selector witness
end ResidualResult27755

namespace ResidualResult27761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27755.actual selector witness +
    ResidualResult27725.actual selector witness
end ResidualResult27761

namespace ResidualResult27771
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27761.actual selector witness *
    ResidualResult27697.actual selector witness
end ResidualResult27771

namespace ResidualResult27774
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 27774
end ResidualResult27774

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
