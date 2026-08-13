/-! Runtime progress markers for long operational-checker evaluations.

The checker itself remains pure and fail-closed.  These markers use `dbg_trace`, which writes to
stderr when Lean evaluates the checker, so generated checker JSON remains the sole stdout output.
They intentionally report bounded node blocks rather than every node in large Tall scopes.
-/

namespace Mxx.Certificate

def operationalProgress
    (phase event scope : String) (processed total : Nat) (detail : String := "") : Bool :=
  let message := "operational_progress phase=" ++ phase ++ " event=" ++ event ++
    " scope=" ++ scope ++ " processed=" ++ toString processed ++
    " total=" ++ toString total ++
    (if detail.isEmpty then "" else " detail=" ++ detail)
  dbg_trace message
  true

def operationalProgressBlock (node total : Nat) : Bool :=
  node < 32 || node + 1 == total || (node + 1) % 1024 == 0

/-- Emit one structured checker diagnostic.  Callers invoke this only on fail-closed paths; the
stable category key makes the message selectable without coupling diagnostics to a protocol. -/
def operationalDiagnostic (category detail : String) : Bool :=
  dbg_trace ("operational_diagnostic category=" ++ category ++ " " ++ detail)
  true

end Mxx.Certificate
