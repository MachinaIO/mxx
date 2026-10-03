import MxxRuntime

namespace Backend

noncomputable section

def backend : MxxRuntime.BackendContext where
  regularLayout _ _ :=
    none

end
end Backend
