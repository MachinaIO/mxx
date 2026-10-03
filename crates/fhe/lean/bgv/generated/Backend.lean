import MxxRuntime

namespace Backend

noncomputable section

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    none

end
end Backend
