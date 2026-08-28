import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult65315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65309.actual selector witness +
    ResidualResult6444.actual selector witness
end ResidualResult65315

namespace ResidualResult65323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65315.actual selector witness *
    ResidualResult3089.actual selector witness
end ResidualResult65323

namespace ResidualResult65328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3089.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult65328

namespace ResidualResult65333
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult6498.actual selector witness
end ResidualResult65333

namespace ResidualResult65337
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65333.actual selector witness -
    ResidualResult65328.actual selector witness
end ResidualResult65337

namespace ResidualResult65343
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65337.actual selector witness +
    ResidualResult6490.actual selector witness
end ResidualResult65343

namespace ResidualResult65353
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65343.actual selector witness *
    ResidualResult6487.actual selector witness
end ResidualResult65353

namespace ResidualResult65359
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65353.actual selector witness +
    ResidualResult65323.actual selector witness
end ResidualResult65359

namespace ResidualResult65369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65359.actual selector witness *
    ResidualResult65290.actual selector witness
end ResidualResult65369

namespace ResidualResult65372
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65372
end ResidualResult65372

namespace ResidualResult65376
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65376
end ResidualResult65376

namespace ResidualResult65381
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult6550.actual selector witness
end ResidualResult65381

namespace ResidualResult65387
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65381.actual selector witness +
    ResidualResult6548.actual selector witness
end ResidualResult65387

namespace ResidualResult65465
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65465
end ResidualResult65465

namespace ResidualResult65468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65468
end ResidualResult65468

namespace ResidualResult65473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65468.actual selector witness *
    ResidualResult65465.actual selector witness
end ResidualResult65473

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
