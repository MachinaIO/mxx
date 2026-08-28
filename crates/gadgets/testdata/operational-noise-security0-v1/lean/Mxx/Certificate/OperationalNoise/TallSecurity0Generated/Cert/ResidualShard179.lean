import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard178

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult23364
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult934.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult23364

namespace ResidualResult23369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult8476.actual selector witness
end ResidualResult23369

namespace ResidualResult23373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23369.actual selector witness -
    ResidualResult23364.actual selector witness
end ResidualResult23373

namespace ResidualResult23379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23373.actual selector witness +
    ResidualResult8468.actual selector witness
end ResidualResult23379

namespace ResidualResult23387
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23379.actual selector witness *
    ResidualResult937.actual selector witness
end ResidualResult23387

namespace ResidualResult23392
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult937.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult23392

namespace ResidualResult23397
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult8517.actual selector witness
end ResidualResult23397

namespace ResidualResult23401
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23397.actual selector witness -
    ResidualResult23392.actual selector witness
end ResidualResult23401

namespace ResidualResult23407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23401.actual selector witness +
    ResidualResult8509.actual selector witness
end ResidualResult23407

namespace ResidualResult23417
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23407.actual selector witness *
    ResidualResult8506.actual selector witness
end ResidualResult23417

namespace ResidualResult23423
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23417.actual selector witness +
    ResidualResult23387.actual selector witness
end ResidualResult23423

namespace ResidualResult23433
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23423.actual selector witness *
    ResidualResult23359.actual selector witness
end ResidualResult23433

namespace ResidualResult23436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23436
end ResidualResult23436

namespace ResidualResult23440
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23440
end ResidualResult23440

namespace ResidualResult23518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23518
end ResidualResult23518

namespace ResidualResult23521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23521
end ResidualResult23521

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
