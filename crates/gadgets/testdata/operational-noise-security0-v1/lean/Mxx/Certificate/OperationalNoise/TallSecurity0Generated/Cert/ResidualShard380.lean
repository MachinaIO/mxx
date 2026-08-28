import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult52606
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 52606
end ResidualResult52606

namespace ResidualResult52609
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 52609
end ResidualResult52609

namespace ResidualResult52614
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2430.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult52614

namespace ResidualResult52619
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult8476.actual selector witness
end ResidualResult52619

namespace ResidualResult52623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52619.actual selector witness -
    ResidualResult52614.actual selector witness
end ResidualResult52623

namespace ResidualResult52629
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52623.actual selector witness +
    ResidualResult8468.actual selector witness
end ResidualResult52629

namespace ResidualResult52637
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52629.actual selector witness *
    ResidualResult2433.actual selector witness
end ResidualResult52637

namespace ResidualResult52642
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2433.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult52642

namespace ResidualResult52647
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult8517.actual selector witness
end ResidualResult52647

namespace ResidualResult52651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52647.actual selector witness -
    ResidualResult52642.actual selector witness
end ResidualResult52651

namespace ResidualResult52657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52651.actual selector witness +
    ResidualResult8509.actual selector witness
end ResidualResult52657

namespace ResidualResult52667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52657.actual selector witness *
    ResidualResult8506.actual selector witness
end ResidualResult52667

namespace ResidualResult52673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52667.actual selector witness +
    ResidualResult52637.actual selector witness
end ResidualResult52673

namespace ResidualResult52683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult52673.actual selector witness *
    ResidualResult52609.actual selector witness
end ResidualResult52683

namespace ResidualResult52686
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 52686
end ResidualResult52686

namespace ResidualResult52690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 52690
end ResidualResult52690

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
