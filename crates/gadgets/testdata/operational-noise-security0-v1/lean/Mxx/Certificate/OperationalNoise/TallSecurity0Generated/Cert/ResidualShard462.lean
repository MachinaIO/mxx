import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard461

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult64718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 64718
end ResidualResult64718

namespace ResidualResult64723
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64718.actual selector witness *
    ResidualResult64716.actual selector witness
end ResidualResult64723

namespace ResidualResult64726
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 64726
end ResidualResult64726

namespace ResidualResult64730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64726.actual selector witness -
    ResidualResult64723.actual selector witness
end ResidualResult64730

namespace ResidualResult64738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64730.actual selector witness *
    ResidualResult64707.actual selector witness
end ResidualResult64738

namespace ResidualResult64741
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 64741
end ResidualResult64741

namespace ResidualResult64746
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64718.actual selector witness *
    ResidualResult64741.actual selector witness
end ResidualResult64746

namespace ResidualResult64749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 64749
end ResidualResult64749

namespace ResidualResult64753
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64749.actual selector witness -
    ResidualResult64746.actual selector witness
end ResidualResult64753

namespace ResidualResult64757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64753.actual selector witness -
    ResidualResult64738.actual selector witness
end ResidualResult64757

namespace ResidualResult64766
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult64595.actual selector witness
end ResidualResult64766

namespace ResidualResult64773
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64766.actual selector witness +
    ResidualResult64588.actual selector witness
end ResidualResult64773

namespace ResidualResult64783
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64773.actual selector witness *
    ResidualResult5859.actual selector witness
end ResidualResult64783

namespace ResidualResult64788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult64788

namespace ResidualResult64793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult5873.actual selector witness
end ResidualResult64793

namespace ResidualResult64797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64793.actual selector witness -
    ResidualResult64788.actual selector witness
end ResidualResult64797

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
