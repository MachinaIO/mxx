import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard528
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard529

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult73702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73702
end ResidualResult73702

namespace ResidualResult73708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73708
end ResidualResult73708

namespace ResidualResult73712
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73712
end ResidualResult73712

namespace ResidualResult73715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73715
end ResidualResult73715

namespace ResidualResult73720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73715.actual selector witness *
    ResidualResult73712.actual selector witness
end ResidualResult73720

namespace ResidualResult73724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73720.actual selector witness -
    ResidualResult73697.actual selector witness
end ResidualResult73724

namespace ResidualResult73732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73724.actual selector witness *
    ResidualResult73681.actual selector witness
end ResidualResult73732

namespace ResidualResult73735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73735
end ResidualResult73735

namespace ResidualResult73740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73692.actual selector witness *
    ResidualResult73735.actual selector witness
end ResidualResult73740

namespace ResidualResult73743
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73743
end ResidualResult73743

namespace ResidualResult73747
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73743.actual selector witness -
    ResidualResult73740.actual selector witness
end ResidualResult73747

namespace ResidualResult73751
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73747.actual selector witness -
    ResidualResult73732.actual selector witness
end ResidualResult73751

namespace ResidualResult73760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult73581.actual selector witness
end ResidualResult73760

namespace ResidualResult73767
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73760.actual selector witness +
    ResidualResult73574.actual selector witness
end ResidualResult73767

namespace ResidualResult73777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult73767.actual selector witness *
    ResidualResult73490.actual selector witness
end ResidualResult73777

namespace ResidualResult73780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73780
end ResidualResult73780

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
