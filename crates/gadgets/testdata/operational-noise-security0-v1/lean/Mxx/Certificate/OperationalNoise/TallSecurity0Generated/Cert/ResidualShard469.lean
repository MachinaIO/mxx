import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard468

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult65741
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65713.actual selector witness *
    ResidualResult65736.actual selector witness
end ResidualResult65741

namespace ResidualResult65744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65744
end ResidualResult65744

namespace ResidualResult65748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65744.actual selector witness -
    ResidualResult65741.actual selector witness
end ResidualResult65748

namespace ResidualResult65752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65748.actual selector witness -
    ResidualResult65733.actual selector witness
end ResidualResult65752

namespace ResidualResult65761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult65590.actual selector witness
end ResidualResult65761

namespace ResidualResult65768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65761.actual selector witness +
    ResidualResult65583.actual selector witness
end ResidualResult65768

namespace ResidualResult65775
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65775
end ResidualResult65775

namespace ResidualResult65778
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65778
end ResidualResult65778

namespace ResidualResult65785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65785
end ResidualResult65785

namespace ResidualResult65788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65788
end ResidualResult65788

namespace ResidualResult65793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3109.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult65793

namespace ResidualResult65798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult6973.actual selector witness
end ResidualResult65798

namespace ResidualResult65802
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65798.actual selector witness -
    ResidualResult65793.actual selector witness
end ResidualResult65802

namespace ResidualResult65808
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65802.actual selector witness +
    ResidualResult6965.actual selector witness
end ResidualResult65808

namespace ResidualResult65816
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65808.actual selector witness *
    ResidualResult3112.actual selector witness
end ResidualResult65816

namespace ResidualResult65821
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3112.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult65821

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
