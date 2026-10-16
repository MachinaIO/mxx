#!/usr/bin/env python3
"""Generate claim-level state, requirement and final assembly for dio2.

All package-specific input tuples, dimensions and source identities are read
from dio2_meta metadata and its fingerprinted export. No finite family is
enumerated. --output-dir supports source-only production checks in scratch.
The final assembly keeps both transition obligations explicit unless both
--step0-theorem and --step1-theorem name completed declarations. The optional
--step0-module and --step1-module select their imports. Generation does not
build any package or alter its lakefile roots.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import dio2_meta as meta


GENERIC = '''/-- Both checks in a successful arithmetic conjunction hold. -/
theorem both {left right : Prop} [Decidable left] [Decidable right]
    (h : ((if left then (1 : Int) else 0) + (if right then (1 : Int) else 0)) = 2) :
    left ∧ right := by
  by_cases hl : left <;> by_cases hr : right <;> simp_all

/-- A bounded nonnegative exponent agrees with its natural monomial. -/
theorem message_one_nat {n : Nat} {exponent : Int}
    (he : 0 ≤ exponent ∧ exponent < 2 * n) :
    message (n := n) 1 exponent = Xe (n := n) exponent.toNat :=
  message_one rfl (Int.toNat_of_nonneg he.1).symm (by omega)

/-- The signed requirement equation identifies the integer plaintext exactly.
The equation itself proves the target range; there is no guessed exponent bound. -/
theorem message_signed {n : Nat} {coefficient exponent target : Int}
    (hc : coefficient * coefficient = 1)
    (he : 0 ≤ exponent ∧ exponent < n)
    (ht : exponent * 2 + (1 - coefficient) * n = target * 2) :
    message (n := n) coefficient exponent = message (n := n) 1 target := by
  have hsign : coefficient = 1 ∨ coefficient = -1 := by
    rcases Int.mul_eq_one_iff_eq_one_or_neg_one.mp hc with hs | hs <;> simp [hs.1]
  rcases hsign with rfl | rfl
  · have hsame : exponent = target := by omega
    rw [hsame]
  · have htRange : 0 ≤ target ∧ target < 2 * n := by omega
    rw [message_one_nat htRange, message,
      Int.emod_eq_of_lt (by omega) (by omega)]
    have hsum : target.toNat = n + exponent.toNat := by omega
    rw [hsum]
    show _ = AdjoinRoot.root (negacyclicModulus n Int) ^ (n + exponent.toNat)
    rw [pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

/-- An arbitrary finite zero/one family of sum one selects exactly one entry. -/
theorem integer_onehot {width : Nat} (value : Fin width → Int)
    (hbound : ∀ digit, 0 ≤ value digit ∧ value digit ≤ 1)
    (hsum : ∑ digit, value digit = 1) :
    ∃ selected : Fin width, ∀ digit, value digit = if digit = selected then 1 else 0 := by
  classical
  have hexists : ∃ selected, value selected = 1 := by
    by_contra hnone
    push_neg at hnone
    have hzero : ∀ digit, value digit = 0 := by
      intro digit
      have := hbound digit
      have := hnone digit
      omega
    simp [hzero] at hsum
  obtain ⟨selected, hselected⟩ := hexists
  refine ⟨selected, fun digit ↦ ?_⟩
  split
  · rename_i hequal
    rw [hequal, hselected]
  · rename_i hdifferent
    by_contra hnonzero
    have hone : value digit = 1 := by have := hbound digit; omega
    have hpair : ∑ index ∈ ({digit, selected} : Finset (Fin width)), value index ≤
        ∑ index, value index :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun index _ _ ↦ (hbound index).1)
    rw [Finset.sum_pair hdifferent, hone, hselected, hsum] at hpair
    omega

/-- One-hot public selectors, quantified over both members and arbitrary bucket width. -/
theorem onehot {n count width : Nat} (coefficient exponent : Fin (count * width) → Int)
    (hc : ∀ member, 0 ≤ coefficient member ∧ coefficient member ≤ 1)
    (he : ∀ member, 0 ≤ exponent member ∧ exponent member ≤ 0)
    (hsum : ∀ member < count, (0 + ∑ digit ∈ Finset.range width,
      famAt coefficient ((Int.ofNat member) * width + (Int.ofNat digit)).toNat) = 1) :
    ∃ selected : Fin count → Fin width, ∀ member digit,
      message (n := n) (coefficient (cellAt member digit)) (exponent (cellAt member digit)) =
        if digit = selected member then (1 : ErrorPoly n) else 0 := by
  classical
  have hindex (member : Fin count) (digit : Fin width) :
      famAt coefficient ((Int.ofNat member.val) * width + (Int.ofNat digit.val)).toNat =
        coefficient (cellAt member digit) := by
    have hcast : ((Int.ofNat member.val) * width + (Int.ofNat digit.val)).toNat =
        width * member.val + digit.val := by
      change ((member.val : Int) * (width : Int) + (digit.val : Int)).toNat = _
      rw [← Nat.cast_mul, ← Nat.cast_add, Int.toNat_natCast, Nat.mul_comm]
    rw [hcast]
    exact famAt_of_lt coefficient (cellAt member digit).isLt
  have hfinite (member : Fin count) : ∑ digit : Fin width, coefficient (cellAt member digit) = 1 := by
    have hs := hsum member.val member.isLt
    rw [zero_add, ← Fin.sum_univ_eq_sum_range] at hs
    simpa only [hindex] using hs
  have hexists (member : Fin count) := integer_onehot
    (fun digit ↦ coefficient (cellAt member digit)) (fun digit ↦ hc _) (hfinite member)
  choose selected hselected using hexists
  refine ⟨selected, fun member digit ↦ ?_⟩
  have hexponent : exponent (cellAt member digit) = 0 := by
    have hbound := he (cellAt member digit)
    omega
  have hmessage : message (n := n) (coefficient (cellAt member digit))
      (exponent (cellAt member digit)) =
      ((coefficient (cellAt member digit) : Int) : ErrorPoly n) := by
    rw [hexponent, message]
    simp
  have hvalue : coefficient (cellAt member digit) =
      if digit = selected member then 1 else 0 := hselected member digit
  rw [hmessage, hvalue]
  split_ifs <;> simp

'''


def projection(base, index, count):
    return base + '.2' * index + ('.1' if index < count - 1 else '')


def initial_result(package, theorem, marker, stage):
    filename = 'Dio2InitialTargets.lean' if 'rel_' in theorem else 'Dio2Initial.lean'
    source = (package / 'proof' / filename).read_text()
    declaration = source.split('theorem ' + theorem + ' ', 1)[1]
    result = marker.strip() + declaration.split(marker, 1)[1].split(' :=', 1)[0]
    inputs = meta.flat_tuple(stage['inputs'])
    result = re.sub(r'\binputs((?:\.2)*\.1)',
                    lambda match: inputs[match[1].count('.2')], result)
    result = re.sub(r'\boutputs\b', f'execution.«{stage["field"]}»', result)
    result = re.sub(r'\bmodulus_0\b', 'W_Stage_initial.modulus_0', result)
    result = re.sub(r'\bsm\b', 'secret.mask', result)
    return re.sub(r'\btp\b', 'secret.payload', result)


def configuration(package, data):
    for relative, digest in data['sources'].items():
        meta.require(hashlib.sha256((package / relative).read_bytes()).hexdigest() == digest,
                     f'export changed since metadata extraction: {relative}')
    view = data['views']['worst']
    claim = (package / 'worst/W_Claim.lean').read_text()
    clauses = claim.split('def Runs ', 1)[1].split(': Prop :=', 1)[1].strip().split(' ∧\n')
    stages = {}
    for index, clause in enumerate(clauses):
        match = re.search(r'(W_Stage_\w+)\.generatedRoot (.*?) (stage_\d+_params) (.*) execution\.«(stage_\d+)»', clause)
        if match is None:
            match = re.search(r'(W_Stage_\w+)\.generatedRoot ()(stage_\d+_params) (.*) execution\.«(stage_\d+)»', clause)
        if match:
            module, arguments, params, inputs, field = match.groups()
            stages[module] = dict(module=module, arguments=arguments, params=params,
                                  inputs=inputs, field=field,
                                  proof=projection('hrun', index, len(clauses)))
    initial = stages['W_Stage_initial']
    initial_args = meta.flat_tuple(initial['inputs'])
    initial_name = meta.one(re.findall(r'external\.(input_\d+)', initial_args[-1]), 'initial state')
    selectors = []
    for step in range(2):
        pre = stages[f'W_Stage_preprocess_{step}_{step}']
        args = meta.flat_tuple(pre['inputs'])
        selectors.append((args[3], args[4]))
    return view, stages, initial_name, selectors


def state_interface(package, data):
    view, stages, initial_name, selectors = configuration(package, data)
    dimension = view['ring_dimension']
    lines = ['import Dio2InitialTargets', 'import Dio2Import', 'import Dio2Correct',
             'import W_Stage_preprocess_0_0_Facts', 'import W_Stage_online_0_Facts', '',
             '/-! Shared additive interface owned by worker c. State0 stores the actual',
             'initial/import/preprocessing/online witnesses, sampled secret lifts and the',
             'state plaintext identity. InitialRows and InitialTargets derive every root',
             'and helper relation from this context; no transition is assumed.',
             'Label is the imported public state-selector plaintext, with gain one.',
             'Generated by gen_c.py; family members remain universally quantified. -/', '',
             'set_option maxRecDepth 16384', 'set_option maxHeartbeats 4000000', '',
             'namespace Dio2Proof', '', 'open Mxx.Primitives MxxRuntime MxxExponentLut',
             '', 'namespace Claim0', '']
    roles = [('initial', 'W_Stage_initial'), ('preprocess', 'W_Stage_preprocess_0_0'),
             ('imported', 'W_Stage_import_0'), ('online', 'W_Stage_online_0')]
    for role, module in roles:
        stage = stages[module]
        args = stage['arguments'].replace('hashModel', 'model')
        lines += [f'noncomputable def {role}Inputs (external : GeneratedClaim.ExternalInputs)',
                  '    (execution : GeneratedClaim.Execution) :=',
                  '  ' + stage['inputs'], '',
                  f'theorem {role}_runs {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}',
                  '    {execution : GeneratedClaim.Execution}',
                  '    (hrun : W_GeneratedClaim.Runs model external execution) :',
                  f'    {module}.generatedRoot {args} W_GeneratedClaim.{stage["params"]}',
                  f'      ({role}Inputs external execution) execution.«{stage["field"]}» :=',
                  f'  {stage["proof"]}', '']
    lines += ['structure Context (model : HashModel) (external : GeneratedClaim.ExternalInputs)',
              '    (execution : GeneratedClaim.Execution) where']
    for role, module in roles:
        stage = stages[module]
        args = stage['arguments'].replace('hashModel', 'model')
        lines += [f'  {role} : {module}.generatedRoot.Ctx {args} W_GeneratedClaim.{stage["params"]}',
                  f'  {role}_inputs : {role}.inputs = {role}Inputs external execution',
                  f'  {role}_facts : {module}.generatedRoot.Facts {role} execution.«{stage["field"]}»']
    lines += ['', 'end Claim0', '',
              'structure State0Witness (model : HashModel) (external : GeneratedClaim.ExternalInputs)',
              f'    (execution : GeneratedClaim.Execution) (value : ErrorPoly {dimension}) where',
              '  context : Claim0.Context model external execution',
              f'  mask : ErrorPoly {dimension}', f'  payload : ErrorPoly {dimension}',
              f'  mask_eq : execution.«{stages["W_Stage_mask"]["field"]}» 0 0 =',
              f'    reducePoly W_Stage_online_0.modulus_0 {dimension} mask',
              f'  payload_eq : execution.«{stages["W_Stage_payload"]["field"]}» 0 0 =',
              f'    reducePoly W_Stage_online_0.modulus_0 {dimension} payload',
              '  mask_small : polyNorm mask ≤ 1', '  payload_small : polyNorm payload ≤ 1',
              f'  state_eq : message 1 (2 * external.{initial_name}) = value', '',
              'def State0 (model : HashModel) (external : GeneratedClaim.ExternalInputs)',
              f'    (execution : GeneratedClaim.Execution) (value : ErrorPoly {dimension}) : Prop :=',
              '  Nonempty (State0Witness model external execution value)', '',
              'noncomputable def stateSelector (external : GeneratedClaim.ExternalInputs)',
              f'    (step : Fin 2) : ErrorPoly {dimension} :=',
              f'  if step = 0 then message ({selectors[0][0]} 0) ({selectors[0][1]} 0)',
              f'  else message ({selectors[1][0]} 0) ({selectors[1][1]} 0)', '',
              'def Label (_model : HashModel) (external : GeneratedClaim.ExternalInputs)',
              f'    (_execution : GeneratedClaim.Execution) (step : Fin 2) (value : ErrorPoly {dimension}) : Prop :=',
              '  stateSelector external step = value ∧ SmulGain value 1', '',
              'end Dio2Proof', '']
    return '\n'.join(lines)


def run_facts(package, data):
    view, stages, initial_name, selectors = configuration(package, data)
    dimension = view['ring_dimension']
    claim = (package / 'worst/W_Claim.lean').read_text()
    clauses = claim.split('def Runs ', 1)[1].split(': Prop :=', 1)[1].strip().split(' ∧\n')
    requirement_index, requirement_clause = meta.one(
        ((index, clause) for index, clause in enumerate(clauses)
         if 'W_Requirement_0.generatedRoot' in clause), 'requirement clause')
    match = re.search(r'W_Requirement_0.generatedRoot (\w+) (.*) execution\.«(\w+)»', requirement_clause)
    requirement_params, requirement_inputs, requirement_output = match.groups()
    requirement_args = meta.flat_tuple(requirement_inputs)
    meta.require(len(requirement_args) >= 11, 'unsupported requirement argument shape')
    generated = (package / 'generated/Claim.lean').read_text()
    validity = generated.split('def ValidExternals ', 1)[1].split(': Prop :=', 1)[1].split('\nstructure Execution', 1)[0]
    conditions = [part.strip() for part in validity.split(' ∧\n')]

    def valid_projection(name):
        index = meta.one((index for index, clause in enumerate(conditions)
                          if name + ' ' in clause or name + ' ∧' in clause), f'validity of {name}')
        return projection('(valid hrun)', index, len(conditions))

    initial = stages['W_Stage_initial']
    mask = stages['W_Stage_mask']
    payload = stages['W_Stage_payload']
    facts = (package / 'proof/Dio2Requirement.lean').read_text()
    root_conditions = re.findall(r'^  (cond_\d+) : (.*)$', facts, re.M)
    meta.require(len(root_conditions) == 2, 'unsupported state-selector requirement shape')
    lines = ['import Dio2State0', 'import Dio2Requirement', '',
             '/-! Claim-level contracts generated by gen_c.py. All sampled secret lifts,',
             'public payload identities and one-hot conditions come from actual Runs.',
             'No numerical decoding margin or online transition is a hypothesis. -/', '',
             'set_option maxRecDepth 16384', 'set_option maxHeartbeats 4000000', '',
             'namespace Dio2Proof.Complete', '',
             'open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen',
             '  RefreshTopProof.GenIface Mxx.ExponentLut.Refresh', '',
             'variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}',
             '  {execution : GeneratedClaim.Execution}', '',
             f'theorem layout_some : Backend.backend.regularLayout W_Stage_online_0.modulus_0 {dimension}',
             '    = some Backend.layout0 := by', '  unfold Backend.backend', '  simp', '',
             'theorem layout_exact : Backend.layout0.droppedModuli = 0 := rfl', '',
             'theorem valid (hrun : W_GeneratedClaim.Runs model external execution) :',
             '    GeneratedClaim.ValidExternals external := hrun.1', '']
    for name, stage in [('mask', mask), ('payload', payload)]:
        lines += [f'theorem {name}_lift (hrun : W_GeneratedClaim.Runs model external execution) :',
                  f'    ∃ value : ErrorPoly {dimension}, execution.«{stage["field"]}» 0 0 =',
                  f'      reducePoly W_Stage_online_0.modulus_0 {dimension} value ∧ polyNorm value ≤ 1 := by',
                  f'  obtain ⟨sample, ⟨_, witness, hsample, hbound⟩, hout⟩ := {stage["proof"]}',
                  '  refine ⟨witness 0 0, ?_, polyNorm_le_of_coeff fun index ↦ ?_⟩',
                  '  · rw [hout, hsample]', '    rfl',
                  '  · have hcoeff := hbound 0 0 index', '    omega', '']
        lines += [f'theorem {name}_matrix_eq {{value : ErrorPoly {dimension}}}',
                  f'    (hvalue : execution.«{stage["field"]}» 0 0 =',
                  f'      reducePoly W_Stage_online_0.modulus_0 {dimension} value) :',
                  f'    execution.«{stage["field"]}» =',
                  f'      fun _ _ ↦ reducePoly W_Stage_online_0.modulus_0 {dimension} value := by',
                  '  funext row column',
                  '  have hrow : row = 0 := Subsingleton.elim _ _',
                  '  have hcolumn : column = 0 := Subsingleton.elim _ _',
                  '  subst row', '  subst column', '  exact hvalue', '']
    lines += ['theorem requirement (hrun : W_GeneratedClaim.Runs model external execution) :',
              '    Dio2Requirement.Facts ' + ' '.join(requirement_args) + ' :=',
              f'  Dio2Requirement.requirement_facts {projection("hrun", requirement_index, len(clauses))} hrun.2' +
              '.2' * (len(clauses) - 2), '', GENERIC]
    roles = ['initial', 'preprocess', 'imported', 'online']
    lines += ['theorem contexts (hrun : W_GeneratedClaim.Runs model external execution) :',
              '    Nonempty (Claim0.Context model external execution) := by']
    for role in roles:
        lines += [f'  obtain ⟨{role}Witness, {role}Body⟩ := Claim0.{role}_runs hrun']
    lines += ['  exact ⟨{']
    for role in roles:
        lines += [f'    {role} := ⟨_, {role}Witness⟩', f'    {role}_inputs := rfl',
                  f'    {role}_facts := {stages[{"initial":"W_Stage_initial", "preprocess":"W_Stage_preprocess_0_0", "imported":"W_Stage_import_0", "online":"W_Stage_online_0"}[role]]["module"]}.generatedRoot.facts ⟨_, {role}Witness⟩ {role}Body']
    lines += ['  }⟩', '',
              'theorem initial (hrun : W_GeneratedClaim.Runs model external execution) :',
              f'    State0 model external execution (Xe (n := {dimension}) (2 * external.{initial_name}.toNat)) := by',
              '  obtain ⟨context⟩ := contexts hrun',
              '  obtain ⟨mask, hmask, hmaskSmall⟩ := mask_lift hrun',
              '  obtain ⟨payload, hpayload, hpayloadSmall⟩ := payload_lift hrun',
              '  refine ⟨⟨context, mask, payload, hmask, hpayload, hmaskSmall, hpayloadSmall, ?_⟩⟩',
              f'  have hbit := {valid_projection("external." + initial_name)}',
              '  rw [message_one_nat (by omega)]', '  congr 1', '  omega', '']
    columns = view['columns']
    base = meta.one({gadget['base'] for gadget in view['gadgets']
                     if any(use['stage'] == 'initial' for use in gadget['uses'])},
                    'initial gadget base')
    lines += ['theorem state0_gadget (hrun : W_GeneratedClaim.Runs model external execution) :',
              f'    ∃ G : ExactMatrix W_Stage_initial.modulus_0 {dimension} 1 {columns},',
              f'      gadgetMatrixRuns Backend.backend {base} {columns} G :=',
              '  initial_gadget (Claim0.initial_runs hrun)', '']
    for name, theorem, marker in [('hint', 'initial_hint', '\n    ∃ error'),
                                  ('encoded_state', 'initial_rel_row_dio2_state', '\n    Rel sm')]:
        result = initial_result(package, theorem, marker, initial)
        lines += [f'theorem state0_{name} (hrun : W_GeneratedClaim.Runs model external execution)',
                  f'    {{value : ErrorPoly {dimension}}}',
                  '    (secret : State0Witness model external execution value)']
        if name == 'encoded_state':
            lines += [f'    {{G : ExactMatrix W_Stage_initial.modulus_0 {dimension} 1 {columns}}}',
                      f'    (hG : gadgetMatrixRuns Backend.backend {base} {columns} G) :']
            proof = 'secret.mask_eq secret.payload_eq hG'
        else:
            lines[-1] += ' :'
            proof = 'secret.mask_eq'
        lines += ['    ' + result + ' := by',
                  '  simpa only [Claim0.initialInputs] using',
                  f'    {theorem} (Claim0.initial_runs hrun) {proof}', '']
    for step, (coefficient, exponent) in enumerate(selectors):
        field = root_conditions[step][0]
        bits = requirement_args[6]
        lines += [f'theorem state_selector{step} (hrun : W_GeneratedClaim.Runs model external execution) :',
                  f'    message (n := {dimension}) ({coefficient} 0) ({exponent} 0) =',
                  f'      Xe (n := {dimension}) (2 * ({bits} {step}).toNat) := by',
                  f'  obtain ⟨hsign, htarget⟩ := both (requirement hrun).{field}',
                  '  simp only [famAt, Fin.isValue, Nat.reduceLT, Nat.zero_lt_one, Nat.one_lt_ofNat,',
                  '    ↓reduceDIte, Fin.zero_eta, Fin.mk_one, decide_eq_true_eq] at hsign htarget',
                  f'  have hexponent := {valid_projection(exponent)} 0',
                  f'  have hbit := {valid_projection(bits)} {step}',
                  f'  rw [message_signed (n := {dimension}) hsign (by omega) (show {exponent} 0 * 2 +',
                  f'    (1 - {coefficient} 0) * {dimension} = ({bits} {step} * 2) * 2 by linarith),',
                  f'    message_one_nat (n := {dimension}) (by omega)]', '  congr 1', '  omega', '']
    lines += ['theorem labels (hrun : W_GeneratedClaim.Runs model external execution) (step : Fin 2) :',
              '    Label model external execution step',
              f'      (Xe (n := {dimension}) (2 * ({requirement_args[6]} step).toNat)) := by',
              '  constructor', '  · fin_cases step',
              '    · exact state_selector0 hrun', '    · exact state_selector1 hrun',
              f'  · exact smulGain_root_pow (by decide : 1 < {dimension}) _', '']
    loop_conditions = re.findall(r'^  (loop_\d+) : ∀ (\w+) < (\d+), (.*)$', facts, re.M)
    label_loops = [entry for entry in loop_conditions if 'if (decide' in entry[3]]
    meta.require(len(label_loops) == 2, 'unsupported root-label requirement loops')
    for step, (field, index, count, condition) in enumerate(label_loops):
        coefficient = requirement_args[2 if step == 0 else 7]
        exponent = requirement_args[3 if step == 0 else 8]
        decisions = [match.end() - 1 for match in re.finditer(r'\(decide \(', condition)]
        meta.require(len(decisions) >= 2, 'requirement is not a signed exponent check')
        proposition = condition[decisions[1] + 1:meta.xmodel.matching(condition, decisions[1])]
        equality = meta.xmodel.split_top(meta.xmodel.strip_parens(proposition), ' = ')
        meta.require(len(equality) == 2, 'unsupported requirement exponent equality')
        factors = meta.xmodel.split_top(meta.xmodel.strip_parens(equality[1]), ' * ')
        meta.require(factors[-1] == '2', 'unsupported signed exponent stride')
        target = factors[0]
        target = re.sub(r'\bx(\d+)\b', lambda match: requirement_args[int(match[1])], target)
        target = re.sub(r'\b' + index + r'\b', 'member.val', target)
        lines += [f'noncomputable def label{step}Exponent (external : GeneratedClaim.ExternalInputs)',
                  f'    (member : Fin {count}) : Int :=', '  ' + target, '',
                  f'theorem label{step}_payload (hrun : W_GeneratedClaim.Runs model external execution)',
                  f'    (member : Fin {count}) :',
                  f'    message (n := {dimension}) ({coefficient} member) ({exponent} member) =',
                  f'      message 1 (label{step}Exponent external member) := by',
                  f'  obtain ⟨hsign, htarget⟩ := both ((requirement hrun).{field} member.val member.isLt)',
                  '  simp only [decide_eq_true_eq, famAt_of_lt _ member.isLt] at hsign htarget',
                  f'  have hexponent := {valid_projection(exponent)} member',
                  f'  apply message_signed (n := {dimension}) hsign (by omega)',
                  f'  simpa only [label{step}Exponent, mul_one, decide_eq_true_eq] using htarget', '',
                  f'theorem label{step}_gain (hrun : W_GeneratedClaim.Runs model external execution)',
                  f'    (member : Fin {count}) :',
                  f'    SmulGain (n := {dimension}) (message ({coefficient} member) ({exponent} member)) 1 := by',
                  f'  rw [label{step}_payload hrun member]',
                  f'  exact smulGain_message_one (by decide : 1 < {dimension}) _', '']
        lines += [f'theorem label{step}_last_payload (hrun : W_GeneratedClaim.Runs model external execution) :',
                  f'    message (n := {dimension}) ({coefficient} ⟨{int(count) - 1}, by decide⟩)',
                  f'      ({exponent} ⟨{int(count) - 1}, by decide⟩) = 1 :=',
                  f'  (label{step}_payload hrun ⟨{int(count) - 1}, by decide⟩).trans (by',
                  f'    simp [label{step}Exponent, MxxIR.selectInt, message])', '']
        imported_stage = stages[f'W_Stage_import_{step}']
        imported_source = (package / 'proof/Dio2Import.lean').read_text()
        payload_match = re.search(rf'theorem import{step}_payload_(\w+).*?\n    (outputs[.\d]+) member =',
                                  imported_source, re.S)
        meta.require(payload_match is not None, 'imported payload interface missing')
        scope, path = payload_match.groups()
        output_path = path.replace('outputs', f'execution.«{imported_stage["field"]}»')
        lines += [f'theorem imported_label{step}_payload (hrun : W_GeneratedClaim.Runs model external execution)',
                  f'    (member : Fin {count}) :',
                  f'    {output_path} member = fun _ _ ↦',
                  f'      reducePoly W_Stage_import_{step}.modulus_0 {dimension}',
                  f'        (message 1 (label{step}Exponent external member)) := by',
                  f'  have hpayload := import{step}_payload_{scope} (import{step}_claim_runs hrun) member',
                  f'  change _ = (fun _ _ ↦ reducePoly W_Stage_import_{step}.modulus_0 {dimension}',
                  f'    (message ({coefficient} member) ({exponent} member))) at hpayload',
                  f'  simpa only [label{step}_payload hrun member] using hpayload', '']
    selector_loops = [entry for entry in loop_conditions
                      if 'Finset.range ' in entry[3] and 'if (decide' not in entry[3]]
    meta.require(len(selector_loops) == len(requirement_args) - 11,
                 'selector loops do not cover the requirement coefficient families')
    for position, (field, index, count, condition) in enumerate(selector_loops):
        coefficient_index = int(meta.one(re.findall(r'famAt x(\d+) ', condition),
                                         'selector coefficient family'))
        coefficient = requirement_args[coefficient_index]
        width = meta.one(re.findall(r'Finset.range (\d+)', condition), 'selector width')
        candidates = []
        for step in range(2):
            inputs = meta.flat_tuple(stages[f'W_Stage_preprocess_{step}_{step}']['inputs'])
            candidates += [inputs[index + 1] for index, value in enumerate(inputs[:-1])
                           if value == coefficient]
        exponent = meta.one(set(candidates), 'selector exponent input')
        lines += [f'theorem selector_{field} (hrun : W_GeneratedClaim.Runs model external execution) :',
                  f'    ∃ selected : Fin {count} → Fin {width}, ∀ member digit,',
                  f'      message (n := {dimension}) ({coefficient} (cellAt member digit))',
                  f'        ({exponent} (cellAt member digit)) =',
                  f'        if digit = selected member then (1 : ErrorPoly {dimension}) else 0 :=',
                  f'  onehot (n := {dimension}) (count := {count}) (width := {width})',
                  f'    {coefficient} {exponent} {valid_projection(coefficient)}',
                  f'    {valid_projection(exponent)} (requirement hrun).{field}', '']
    lines += ['end Dio2Proof.Complete', '']
    return '\n'.join(lines)


def complete_source(package, data, step0=None, step1=None,
                    module0='Dio2Online0Assembly', module1='Dio2Online1Assembly'):
    dimension = data['views']['worst']['ring_dimension']
    fields = []
    for step in range(2):
        target = f'(Xe (n := {dimension}) (2 * (entry table{step} '
        target += f'((state + {"bit" if step == 0 else "1 + bit"} : Nat) : Int)).toNat))'
        conclusion = 'State1 model external execution' if step == 0 else 'FinalOutput execution'
        fields.append((f'step{step}',
            '∀ {model external execution}, W_GeneratedClaim.Runs model external execution →\n'
            '    ∀ (state bit : Nat), state ≤ 1 → bit ≤ 1 →\n'
            f'    Label model external execution {step} (Xe (n := {dimension}) (2 * bit)) →\n'
            f'    State{step} model external execution (Xe (n := {dimension}) (2 * state)) →\n'
            f'    {conclusion}\n      {target}'))
    meta.require((step0 is None) == (step1 is None), 'provide both transition theorems or neither')
    imports = ['Dio2RunFacts', 'Dio2Online1Inputs']
    if step0 is not None:
        for name in (step0, step1, module0, module1):
            meta.require(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.']*", name) is not None,
                         'transition linkage must use Lean declaration/module names')
        imports += [module0, module1]
    lines = [*(f'import {name}' for name in dict.fromkeys(imports)), '',
             '/-! Claim assembly generated by gen_c.py. Initial and label fields are proved',
             'from actual runs in Dio2RunFacts. RemainingTransitions states exactly the two',
             'online-step obligations: no decoded-output or numerical-margin assumption.',
             'Without both supplied step theorems, the assembly is explicitly conditional.',
             'Pass --step0-theorem and --step1-theorem only after those declarations build',
             'to generate the unconditional dio2_correctness_claim. -/', '',
             'namespace Dio2Proof.Complete', '',
             'open Mxx.Primitives MxxRuntime MxxExponentLut', '',
             '/-- Owned by s0 (step0) and s1 (step1), not assumed by the run contracts. -/',
             'structure RemainingTransitions : Prop where']
    lines += [f'  {name} : {statement.strip()}' for name, statement in fields]
    lines += ['', 'def correctInputs (transitions : RemainingTransitions) : CorrectInputs where',
              '  State0 := Dio2Proof.State0', '  State1 := Dio2Proof.State1',
              '  Label := Dio2Proof.Label', '  initial := initial', '  labels := labels',
              '  step0 := transitions.step0', '  step1 := transitions.step1', '',
              'theorem dio2_correctness_claim_of_transitions (transitions : RemainingTransitions) :',
              '    GeneratedClaim.CorrectnessClaim := certificate (correctInputs transitions)', '']
    if step0 is not None:
        lines += ['theorem remainingTransitions : RemainingTransitions where',
                  f'  step0 := {step0}', f'  step1 := {step1}', '',
                  'theorem dio2_correctness_claim : GeneratedClaim.CorrectnessClaim :=',
                  '  certificate (correctInputs remainingTransitions)', '']
    lines += ['end Dio2Proof.Complete', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--step0-theorem')
    parser.add_argument('--step1-theorem')
    parser.add_argument('--step0-module', default='Dio2Online0Assembly')
    parser.add_argument('--step1-module', default='Dio2Online1Assembly')
    args = parser.parse_args()
    data = json.loads(args.metadata.read_text()) if args.metadata else meta.extract(args.package)
    output = args.output_dir or args.package / 'proof'
    files = {'Dio2State0.lean': state_interface(args.package, data),
             'Dio2RunFacts.lean': run_facts(args.package, data),
             'Dio2Complete.lean': complete_source(args.package, data, args.step0_theorem,
                 args.step1_theorem, args.step0_module, args.step1_module)}
    output.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        destination = output / name
        if not destination.exists() or destination.read_text() != source:
            destination.write_text(source)
    print(f'generated {", ".join(files)} in {output}')


if __name__ == '__main__':
    main()
