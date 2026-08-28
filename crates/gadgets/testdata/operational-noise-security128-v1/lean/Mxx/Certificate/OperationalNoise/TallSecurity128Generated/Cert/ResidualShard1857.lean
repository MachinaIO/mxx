import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1758
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1759
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1844
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1846
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1847
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1848
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1850
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1851
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1852
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1854
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1855
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1856

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult265516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265506.actual selector witness *
    ResidualResult15882.actual selector witness
end ResidualResult265516

namespace ResidualResult265521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult251403.actual selector witness
end ResidualResult265521

namespace ResidualResult265526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult251273.actual selector witness *
    ResidualResult15896.actual selector witness
end ResidualResult265526

namespace ResidualResult265530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265526.actual selector witness -
    ResidualResult265521.actual selector witness
end ResidualResult265530

namespace ResidualResult265536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265530.actual selector witness +
    ResidualResult31516.actual selector witness
end ResidualResult265536

namespace ResidualResult265543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265536.actual selector witness -
    ResidualResult265536.actual selector witness
end ResidualResult265543

namespace ResidualResult265548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265543.actual selector witness +
    ResidualResult265516.actual selector witness
end ResidualResult265548

namespace ResidualResult265553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265548.actual selector witness +
    ResidualResult265304.actual selector witness
end ResidualResult265553

namespace ResidualResult265558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265553.actual selector witness +
    ResidualResult265092.actual selector witness
end ResidualResult265558

namespace ResidualResult265563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265558.actual selector witness +
    ResidualResult264880.actual selector witness
end ResidualResult265563

namespace ResidualResult265568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265563.actual selector witness +
    ResidualResult264668.actual selector witness
end ResidualResult265568

namespace ResidualResult265573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265568.actual selector witness +
    ResidualResult264456.actual selector witness
end ResidualResult265573

namespace ResidualResult265578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265573.actual selector witness +
    ResidualResult264244.actual selector witness
end ResidualResult265578

namespace ResidualResult265583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265578.actual selector witness +
    ResidualResult264032.actual selector witness
end ResidualResult265583

namespace ResidualResult265588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265583.actual selector witness +
    ResidualResult263820.actual selector witness
end ResidualResult265588

namespace ResidualResult265593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult265588.actual selector witness +
    ResidualResult263608.actual selector witness
end ResidualResult265593

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
