#!/usr/bin/env python3
"""Generate `proof/Dio2KeyAgree0.lean`: the key agreement behind `hpre` of `online0_part_prime0`.

`online0_part_prime0` (`proof/Dio2Online0PartPrimes.lean`) needs, for refresh part `i` of step 0,
that the preimage `P` (the part's input `w_1`) satisfies
`B_k P = (K_g,k D_top + K₁,k D_bot - K_x,k Y) + K_f,k + K_s,k` with the keys of the online rows.
The preimage solves `B_0 P = K` for member `i` of the keys stage's output family `dio2/key/0`,
whose row `k` the keys stage (`W_Stage_keys_online_0_0`, root `parallel_generatedRoot_1267`)
computes as the same combination over its own row-`k` inputs. The keys stage runs the online graph
on key rows and its own label payloads, so the proof never equates rows or payloads of the two
stages; it shows that both stages compute every needed key from the same hashed root keys, shared
artifacts and digits.

The file has hand-written parts (the template below: the part-level combination, the stage and
claim wrappers) and generated parts, derived from the worst-case exports `worst/W_Stage_online_0`
and `worst/W_Stage_keys_online_0_0` (their `_Facts` modules) and `worst/W_Claim.lean`:

* `Link0`: one field per stage input both stages read from the same `W_Claim` source;
* the block-diagonal digit chains of the online part and the keys stage's prime-0 root;
* deterministic agreements (`Agree`): wire-by-wire equalities of hashes, concatenations,
  reductions, rebases, decompositions, slices of stacked products (rows `1, 2` only, through
  `stack_rows_agree`: they do not read the ciphertext row) and zero-count loops, recursively;
* the PRF chains (`Chain`): static and dynamic PRF cells, row sums, LUT members, placements, member
  and column totals, each through its role key lemma (`PrfCellFlat.keys`, `rowsum_keys`,
  `LutMemberFlat.keys`, `placement_keys`, `monomial_keys`, `member_total_keys`,
  `column_total_keys`) applied in both stages, quantified over members;
* the state LUT loop (`iter_link_gen` with `LutRoundStep.keys_1`);
* the 36 `cₓ` targets (`CX`): the online target theorems' key forms matched against the keys
  stage's two `cₓ` rows.

Usage: gen_key_agree0.py [package dir]   (default: ../average-n8-p5x50 next to this script)
"""
import functools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../../../exponent-lut/lean/refresh_top/generator'))
from xmodel import app_split, split_top, strip_parens, tuple_items  # noqa: E402

PKG = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, '../average-n8-p5x50')
WORST = os.path.join(PKG, 'worst')
OS = 'W_Stage_online_0'
KS = 'W_Stage_keys_online_0_0'


def packlist(F, field):
    """The list of a `familyPack` atom."""
    txt = F[field][1]
    m = re.search(r'\[(.*)\]', txt)
    return [x.strip() for x in m.group(1).split(',')]


# --------------------------------------------------------------------------- claim links

def claim_args(stage):
    """The input tuple items of `stage.generatedRoot` in `W_Claim.Runs`."""
    t = open(os.path.join(WORST, 'W_Claim.lean')).read()
    for line in t.split('\n'):
        if line.strip().startswith(stage + '.generatedRoot'):
            l = line.strip()
            i = l.index('((')
            depth = 0
            for j in range(i, len(l)):
                if l[j] == '(':
                    depth += 1
                elif l[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
            return [x.strip() for x in split_top(strip_parens(l[i + 1:j]), ', ')]


def input_wires(mod):
    """Root input wires of a stage: wire -> input index."""
    t = open(os.path.join(WORST, mod + '.lean')).read()
    body = t[t.index('abbrev generatedRoot.body'):]
    return {m.group(1): m.group(2).count('.2')
            for m in re.finditer(r'^  let (w_\d+_\d+) : .*? := inputs((?:\.2)*)(?:\.1)?$', body, re.M)}


def link_pairs():
    """(online wire, keys wire) for every input both stages read from the same claim source (the
    keys stage's input `k` is the online stage's input `k + 2`)."""
    on, ke = claim_args(OS), claim_args(KS)
    mo, mk = input_wires(OS), input_wires(KS)
    ko = {i: w for w, i in mk.items()}
    pairs = []
    for w, i in sorted(mo.items(), key=lambda x: x[1]):
        kw = ko.get(i - 2)
        if kw is not None and on[i] == ke[i - 2]:
            pairs.append((w, kw))
    return pairs


def link_structure(pairs):
    out = ['/-- The inputs the online stage of step 0 and its keys stage read from the same source (`W_Claim`). -/',
           'structure Link0 {backend : BackendContext} {hashModel : HashModel} {params : W_Stage_online_0.Params}',
           '    {paramsK : W_Stage_keys_online_0_0.Params}',
           '    (Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params)',
           '    (Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK) : Prop where']
    for w, kw in pairs:
        out.append(f'  e{w.split("_")[1]} : W_Stage_online_0.generatedRoot.{w} Con = W_Stage_keys_online_0_0.generatedRoot.{kw} Ck')
    return '\n'.join(out) + '\n\n'


links = {}

# --------------------------------------------------------------------------- facts index

@functools.lru_cache(maxsize=None)
def load(mod, scope='generatedRoot'):
    """The `scope` facts of `<mod>_Facts.lean`: field -> (path, atom text); wire -> let term."""
    t=open(os.path.join(WORST, mod+'_Facts.lean')).read()
    nots=dict(re.findall(r'^local notation "(mxx\w+)" => (.*)$',t,re.M))
    def expand(s):
        return re.sub(r'\bmxx(?:Mat|Nat)\d+\b', lambda m: nots.get(m.group(0),m.group(0)), s)
    fields={}
    cur=None
    for line in t.split('\n'):
        m=re.match(r'^structure '+re.escape(scope)+r'\.Facts_(\d+) ',line)
        if m: cur=m.group(1); continue
        if line.startswith('structure ') or line.startswith('theorem ') or line.startswith('def '): cur=None
        m=re.match(r'^  (\S+) : (.*)$',line)
        if m and cur is not None:
            txt=expand(re.sub(r'\('+re.escape(scope)+r'\.(w_\w+) ctx\)',r'\1',m.group(2)))
            fields[m.group(1)]=('F.c'+cur+'.'+m.group(1),txt)
    lets={}
    for m in re.finditer(r'^def '+re.escape(scope)+r'\.(w_\w+) \{[^\n]*\n  ([^\n]*)',t,re.M):
        lets[m.group(1)]=expand(re.sub(r'\('+re.escape(scope)+r'\.(w_\w+) ctx\)',r'\1',m.group(2)))
    return fields,lets

# --------------------------------------------------------------------------- deterministic agreements

ON='W_Stage_online_0.generatedRoot'
KE='W_Stage_keys_online_0_0.generatedRoot'
UNIQUE={'modulusReduceRuns':'modulusReduceRuns_unique','centeredRebaseRuns':'centeredRebaseRuns_unique',
 'gadgetDecomposeRuns':'gadgetDecomposeRuns_unique','compactCenteredRebaseRuns':'compactCenteredRebaseRuns_unique',
 'concatRows':'concatRows_unique','concatColumns':'concatColumns_unique','sliceMatrix':'sliceMatrix_unique',
 'ringAutomorphismRuns':'ringAutomorphismRuns_unique','familyGetStatic':'familyGetDynamic_unique',
 'familyGetDynamic':'familyGetDynamic_unique'}
WIRE=re.compile(r'\bw_\d+(?:_\d+)?(?:_concat_\d+)?\b')

class Agree:
    """Equalities `online wire = keys wire` for deterministic computations, proved recursively from
    the stage links, the role-free uniqueness lemmas and `given` equalities (loop agreements).
    Every proof line is collected in `lines`; `eq` returns the name of the equality."""
    def __init__(self, links):
        self.Fo,self.Lo=load('W_Stage_online_0')
        self.Fk,self.Lk=load('W_Stage_keys_online_0_0')
        self.links=links   # (wo,wk) -> proof term
        self.done={}       # (wo,wk) -> name
        self.lines=[]
        self.n=0
        self.prod_o=self._producers(self.Fo); self.prod_k=self._producers(self.Fk)
        self.loop_o=self._loops(self.Fo); self.loop_k=self._loops(self.Fk)
        self.given={}
    def _loops(self,F):
        res={}
        for k,(p,txt) in F.items():
            if not txt.startswith('MxxIR.IterRuns'): continue
            h,args=app_split(txt)
            cnt=strip_parens(args[1]); init=args[2]; final=args[3]
            fi=[strip_parens(x) for x in tuple_items(final)]; ii=[strip_parens(x) for x in tuple_items(init)]
            for idx,w in enumerate(fi):
                res[w]=(p,cnt,idx,ii)
        return res
    def _producers(self,F):
        res={}
        for k,(p,txt) in F.items():
            m=re.match(r'^MxxRuntime\.(\w+) (.*)$',txt)
            if not m: continue
            h,args=app_split(txt)
            last=strip_parens(args[-1]) if args else ''
            if WIRE.fullmatch(last): res.setdefault(last,(m.group(1),p,args))
        return res
    def ref(self,side,w): return f'({ON if side=="o" else KE}.{w} Con)' if side=='o' else f'({KE}.{w} Ck)'
    def shape(self,term):
        return WIRE.sub('W',term)
    def eq(self,wo,wk):
        key=(wo,wk)
        if key in self.done: return self.done[key]
        if key in self.links: return self.links[key]
        lo=self.Lo.get(wo,''); lk=self.Lk.get(wk,'')
        name=f'a{self.n}'; self.n+=1
        lhs=self.ref('o',wo); rhs=self.ref('k',wk)
        if lo and not lo.startswith('ctx.') and lk and not lk.startswith('ctx.'):
            # let-defined on both sides: same shape, agree on the argument wires
            if self.shape(lo)!=self.shape(lk): raise Exception(f'let shape {wo}={lo} vs {wk}={lk}')
            wa=WIRE.findall(lo); wb=WIRE.findall(lk)
            subs=[self.eq(a,b) for a,b in zip(wa,wb)]
            subs=list(dict.fromkeys(subs))
            if subs:
                body=f'by unfold {ON}.{wo} {KE}.{wk}; rw [{", ".join(subs)}]'
            else:
                body='rfl'
            self.lines.append(f'  have {name} : {lhs} = {rhs} := {body}')
            self.done[key]=name; return name
        if key in self.given: return self.given[key]
        if lo.startswith('ctx.inputs') or lk.startswith('ctx.inputs'):
            raise Exception(f'input pair without link: {wo} {wk}')
        if wo in self.loop_o and wk in self.loop_k:
            po,co,io,iio=self.loop_o[wo]; pk,ck,ik,iik=self.loop_k[wk]
            if co!='Int.toNat (0)' or ck!='Int.toNat (0)' or io!=ik:
                raise Exception(f'loop output {wo} {wk} ({co},{ck})')
            sub=self.eq(iio[io],iik[ik])
            proj='s'+'.2'*io+'.1'
            self.lines.append(f'  have {name} : {lhs} = {rhs} := (congrArg (fun s ↦ {proj}) (iterRuns_zero_eq Fon.{po[2:]} rfl)).trans ({sub}.trans (congrArg (fun s ↦ {proj}) (iterRuns_zero_eq Fk.{pk[2:]} rfl)).symm)')
            self.done[key]=name; return name
        po=self.prod_o.get(wo); pk=self.prod_k.get(wk)
        if not po or not pk: raise Exception(f'no producer {wo} {wk}')
        if po[0]!=pk[0]: raise Exception(f'kind {wo}:{po[0]} vs {wk}:{pk[0]}')
        kind=po[0]
        ao=po[2][:-1]; ak=pk[2][:-1]
        if kind=='hashSample':
            # model tag comps key: comps may mention wires
            so=' '.join(ao); sk=' '.join(ak)
            if self.shape(so)!=self.shape(sk): raise Exception(f'hash shape {wo} {wk}: {so} vs {sk}')
            subs=[self.eq(a,b) for a,b in zip(WIRE.findall(so),WIRE.findall(sk))]
            subs=list(dict.fromkeys(subs))
            ho=f'h{name}'
            self.lines.append(f'  have {ho} := Fon.{po[1][2:]}')
            if subs: self.lines.append(f'  rw [{", ".join(subs)}] at {ho}')
            self.lines.append(f'  have {name} : {lhs} = {rhs} := (MxxExponentLut.hashSample_eq {ho}).trans (MxxExponentLut.hashSample_eq Fk.{pk[1][2:]}).symm')
            self.done[key]=name; return name
        if kind=='gadgetMatrixRuns':
            self.lines.append(f'  have {name} : {lhs} = {rhs} := gadgetMatrixRuns_unique Fon.{po[1][2:]} Fk.{pk[1][2:]}')
            self.done[key]=name; return name
        if kind=='sliceMatrix' and [x.strip() for x in ao[1:]] in (['1','2','0','25'],['2','3','0','25']) and [x.strip() for x in ak[1:]]==[x.strip() for x in ao[1:]]:
            r=int(ao[1])
            Xo=strip_parens(ao[0]); Xk=strip_parens(ak[0])
            mo=re.match(r'^MxxRuntime\.matrixMul (\S+) (\S+)$',self.Lo.get(Xo,'')); mk=re.match(r'^MxxRuntime\.matrixMul (\S+) (\S+)$',self.Lk.get(Xk,''))
            if mo and mk:
                so=self.stack(self.Fo,self.prod_o,mo.group(1)); sk=self.stack(self.Fk,self.prod_k,mk.group(1))
                if so and sk:
                    es=[self.eq(so[x],sk[x]) for x in ('l1','l2','r1','r2')]
                    ed=self.eq(mo.group(2),mk.group(2))
                    args=' '.join('Fon.'+so['p'+x][2:] for x in ('l01','l','r01','r','both'))+' '+' '.join('Fk.'+sk['p'+x][2:] for x in ('l01','l','r01','r','both'))
                    self.lines.append(f'  have {name} : {lhs} = {rhs} := by')
                    self.lines.append(f'    rw [slice_rowOfG (by norm_num) Fon.{po[1][2:]}, slice_rowOfG (by norm_num) Fk.{pk[1][2:]}]')
                    self.lines.append(f'    exact (stack_rows_agree {args} {" ".join(es)} {ed}).{r}')
                    self.done[key]=name; return name
        if kind in UNIQUE:
            so=' '.join(ao); sk=' '.join(ak)
            if self.shape(so)!=self.shape(sk): raise Exception(f'{kind} shape {wo} {wk}: {so} vs {sk}')
            subs=[self.eq(a,b) for a,b in zip(WIRE.findall(so),WIRE.findall(sk))]
            subs=list(dict.fromkeys(subs))
            ho=f'h{name}'
            self.lines.append(f'  have {ho} := Fon.{po[1][2:]}')
            if subs: self.lines.append(f'  rw [{", ".join(subs)}] at {ho}')
            self.lines.append(f'  have {name} : {lhs} = {rhs} := {UNIQUE[kind]} {ho} Fk.{pk[1][2:]}')
            self.done[key]=name; return name
        raise Exception(f'unsupported {kind} {wo} {wk}')
    def stack(self,F,P,both):
        def at(w,kind):
            if w not in P or P[w][0]!=kind: return None
            return P[w]
        b=at(both,'concatColumns')
        if not b: return None
        l,r=[strip_parens(x) for x in b[2][:2]]
        res={'pboth':b[1]}
        for side,w in (('l',l),('r',r)):
            x=at(w,'concatRows')
            if not x: return None
            w01,w2=[strip_parens(y) for y in x[2][:2]]
            y=at(w01,'concatRows')
            if not y: return None
            w0,w1=[strip_parens(z) for z in y[2][:2]]
            res['p'+side]=x[1]; res['p'+side+'01']=y[1]; res[side+'1']=w1; res[side+'2']=w2
        return res
    def sub(self,term,side):
        return WIRE.sub(lambda m: self.ref(side,m.group(0)), term)


# --------------------------------------------------------------------------- PRF chains
OS='W_Stage_online_0'; KS='W_Stage_keys_online_0_0'

def call_args(txt, scope):
    """The input tuple items of the child call `scope … (inputs) (outputs)` in a root atom."""
    i=txt.index(scope+' ')
    h,args=app_split(txt[i:].rstrip(')'))
    return [strip_parens(x) for x in tuple_items(args[-2])]

def base(x):
    m=re.match(r'^(w_\d+_\d+)(?: i)?$',x.strip())
    return m.group(1) if m else None

def tup(side,w,n,idx=None):
    S=ON if side=='o' else KE; c='Con' if side=='o' else 'Ck'
    items=[f'{S}.w_{n}_{k} {c}'+(f' {idx}' if idx else '') for k in range(6)]
    s='()'
    for it in reversed(items): s=f'({it}, {s})'
    return s

class Chain:
    """The key agreement of a PRF chain, bundle by bundle: `k<n>` states `Enc.A` of the online
    bundle `n`'s outputs equals that of the keys stage's matching bundle, for every member."""
    def __init__(self):
        self.A=Agree(links)
        self.out=[]
        self.fam={}  # online family wire -> name of family equality
    def eqname(self,wo,wk):
        if wo in self.fam: return self.fam[wo]
        return self.A.eq(wo,wk)
    def rewrites(self,args_o,args_k,positions):
        names=[]
        for p in positions:
            a=base(args_o[p]); b=base(args_k[p])
            if a is None or b is None: raise Exception(f'pos {p}: {args_o[p]} {args_k[p]}')
            names.append(self.eqname(a,b))
        return list(dict.fromkeys(names))
    def family_eqs(self,no,nk,count):
        lines=[]
        for c,(a,b) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            nm=f'f{no}_{c}'
            lines.append(f'  have {nm} : {ON}.w_{no}_{c} Con = {KE}.w_{nk}_{c} Ck := funext fun o ↦ congrFun (congrFun (k{no} o) {a}) {b}')
            self.fam[f'w_{no}_{c}']=nm
        return lines
    def cells(self,no,nk,N,fo,fk,dynamic):
        Fo=self.A.Fo; Fk=self.A.Fk
        po,to=Fo[fo]; pk,tk=Fk[fk]
        ao=call_args(to,f'parallel_generatedRoot_{no}'); ak=call_args(tk,f'parallel_generatedRoot_{nk}')
        pos=[0,2,4,5,8,9,11,12,13]
        rw=self.rewrites(ao,ak,pos)
        lem='PrfCellFlat.keys_1' if dynamic else 'PrfCellFlat.keys'
        extra=' (by omega)' if dynamic else ''
        L=[f'  have k{no} : ∀ j : Fin {N}, Enc.A (Tup.ev (k := 6) {tup("o",None,no)} j) =',
           f'      Enc.A (Tup.ev (k := 6) {tup("k",None,nk)} j) := by',
           '    intro j',
           f'    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := Fon.{po[2:]} j',
           f'    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc\'⟩ := Fk.{pk[2:]} j',
           f'    refine ({lem} (by norm_num) hq1 layout hlayout hexact ({OS}.parallel_generatedRoot_{no}.sem hc){extra}).trans',
           f'      (Eq.trans ?_ ({lem} (by norm_num) hq1 layout hlayout hexact ({KS}.parallel_generatedRoot_{nk}.sem hc\'){extra}).symm)',
           f'    rw [{", ".join(rw)}]']
        return L
    def rowsum(self,no,nk,M,fo,fk,prev_o,prev_k):
        po=self.A.Fo[fo][0]; pk=self.A.Fk[fk][0]
        L=[f'  have k{no} : ∀ o : Fin {M}, Enc.A {tup("o",None,no,"o")} = Enc.A {tup("k",None,nk,"o")} := by',
           '    intro o',
           f'    rw [rowsum_keys ({OS}.parallel_generatedRoot_{no}.sem (Fon.{po[2:]} o)),',
           f'      rowsum_keys ({KS}.parallel_generatedRoot_{nk}.sem (Fk.{pk[2:]} o))]',
           f'    exact Finset.sum_congr rfl fun j _ ↦ k{prev_o} _']
        return L
    def lut(self,no,nk,M,fo,fk,childo,childk):
        Fo=self.A.Fo; Fk=self.A.Fk
        po,to=Fo[fo]; pk,tk=Fk[fk]
        ao=call_args(to,f'parallel_generatedRoot_{no}'); ak=call_args(tk,f'parallel_generatedRoot_{nk}')
        rw=self.rewrites(ao,ak,[0,1,2,3,5,8,9,11,12])
        L=[f'  have k{no} : ∀ o : Fin {M}, Enc.A {tup("o",None,no,"o")} = Enc.A {tup("k",None,nk,"o")} := by',
           '    intro o',
           f'    refine (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num) hq1 layout hlayout hexact',
           f'        (fun i a b hb ↦ {OS}.{childo}.sem hb) ({OS}.parallel_generatedRoot_{no}.sem (Fon.{po[2:]} o))).trans',
           f'      (Eq.trans ?_ (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num) hq1 layout hlayout hexact',
           f'        (fun i a b hb ↦ {KS}.{childk}.sem hb) ({KS}.parallel_generatedRoot_{nk}.sem (Fk.{pk[2:]} o))).symm)',
           f'    rw [{", ".join(rw)}]']
        return L
    def placement(self,no,nk,M,fo,fk,prev_o):
        po=self.A.Fo[fo][0]; pk=self.A.Fk[fk][0]
        L=[f'  have k{no} : ∀ o : Fin {M}, Enc.A {tup("o",None,no,"o")} = Enc.A {tup("k",None,nk,"o")} := by',
           '    intro o',
           f'    rw [placement_keys ({OS}.parallel_generatedRoot_{no}.sem (Fon.{po[2:]} o)),',
           f'      placement_keys ({KS}.parallel_generatedRoot_{nk}.sem (Fk.{pk[2:]} o))]',
           f'    exact congrArg _ (k{prev_o} o)']
        return L

def getters(F,fam,idx):
    res={}
    for k,(p,txt) in F.items():
        m=re.match(r'^MxxRuntime\.familyGetStatic (w_\d+)_(\d) \('+str(idx)+r'\) (w_\d+_\d+)$',txt)
        if m and m.group(1)==fam: res[int(m.group(2))]=(m.group(3),p)
    return [res[c] for c in range(6)]

def tensors(F,loop):
    res={}
    for k,(p,txt) in F.items():
        m=re.match(r'^MxxRuntime\.tensorRuns (w_\d+_\d+) '+loop+r'_(\d) (w_\d+_\d+)$',txt)
        if m: res[int(m.group(2))]=(m.group(3),p)
    return [res[c] for c in range(6)]

def tuple_of(S,c,ws):
    s='()'
    for w in reversed(ws): s=f'({S}.{w} {c}, {s})'
    return s

def column(ch,no,nk,famo,famk,memo,memk,colo,colk,C,F,W,fmo,fmk,fco,fck):
    """Column total: fams = placement outputs `famo`/`famk`, member loop `memo`, column loop `colo`."""
    A=ch.A
    L=[]
    for side,Fx,fam,mem,col,fm,fc,S,c,SS,Fn in (('o',A.Fo,famo,memo,colo,fmo,fco,ON,'Con',OS,'Fon'),('k',A.Fk,famk,memk,colk,fmk,fck,KE,'Ck',KS,'Fk')):
        ga=getters(Fx,f'w_{fam}',0); gb=getters(Fx,f'w_{fam}',1)
        te=tensors(Fx,f'w_{mem}')
        fams=tuple_of(S,c,[f'w_{fam}_{k}' for k in range(6)])
        L+= [f'  have hx0{side} := member_prefix (α := ExactMatrix W_Stage_online_0.modulus_0 8 1 25) (k := 6) (F := {F}) (Wm := {W}) (by norm_num) (by norm_num)',
             f'    (fams := {fams})',
             f'    (a := {tuple_of(S,c,[x for x,_ in ga])})',
             f'    (b := {tuple_of(S,c,[x for x,_ in gb])})',
             f'    (fun c ↦ by fin_cases c; exacts [{", ".join(Fn+"."+p[2:] for _,p in ga)}])',
             f'    (fun c ↦ by fin_cases c; exacts [{", ".join(Fn+"."+p[2:] for _,p in gb)}])',
             f'    (fun c ↦ by fin_cases c <;> rfl) {Fn}.{fm}',
             f'    (fun i cur next hb ↦ {SS}.sequential_generatedRoot_{mem}.sem hb)',
             f'  have col{side} := column_total_keys (C := {C}) (m := 25) (W := {W}) (F := {F}) (by norm_num) (by norm_num) hx0{side}',
             f'    (fun c ↦ by fin_cases c; exacts [{", ".join(Fn+"."+p[2:] for _,p in te)}]) {Fn}.{fc}',
             f'    (fun i cur next hb ↦ by have hc := {SS}.sequential_generatedRoot_{col}.sem hb; exact hc)']
    fo=tuple_of(ON,'Con',[f'w_{colo}_{k}' for k in range(6)])
    fk=tuple_of(KE,'Ck',[f'w_{colk}_{k}' for k in range(6)])
    L+=[f'  have kcol : ∀ cc : Fin {C}, Enc.A (blkT (r := 1 * 1) (C := {C}) (m := 25) {fo} cc) =',
        f'      Enc.A (blkT (r := 1 * 1) (C := {C}) (m := 25) {fk} cc) := by',
        '    intro cc',
        '    rw [colo cc, colk cc]',
        f'    exact Finset.sum_congr rfl fun j _ ↦ k{famo} _']
    return L

def monomial(ch,no,nk,M,fo,fk,prev_o):
    po=ch.A.Fo[fo][0]; pk=ch.A.Fk[fk][0]
    return [f'  have k{no} : ∀ o : Fin {M}, Enc.A {tup("o",None,no,"o")} = Enc.A {tup("k",None,nk,"o")} := by',
           '    intro o',
           f'    rw [monomial_keys ({OS}.parallel_generatedRoot_{no}.sem (Fon.{po[2:]} o)),',
           f'      monomial_keys ({KS}.parallel_generatedRoot_{nk}.sem (Fk.{pk[2:]} o))]',
           f'    exact congrArg _ (k{prev_o} o)']

def member_total(ch,famo,famk,memo,memk,M,fmo,fmk):
    A=ch.A
    L=[]
    for side,Fx,fam,mem,fm,S,c,SS,Fn in (('o',A.Fo,famo,memo,fmo,ON,'Con',OS,'Fon'),('k',A.Fk,famk,memk,fmk,KE,'Ck',KS,'Fk')):
        ga=getters(Fx,f'w_{fam}',0); gb=getters(Fx,f'w_{fam}',1)
        fams=tuple_of(S,c,[f'w_{fam}_{k}' for k in range(6)])
        L+= [f'  have tot{side} := member_total_keys (M := {M}) (by norm_num)',
             f'    (fams := {fams})',
             f'    (a := {tuple_of(S,c,[x for x,_ in ga])})',
             f'    (b := {tuple_of(S,c,[x for x,_ in gb])})',
             f'    (fun c ↦ by fin_cases c; exacts [{", ".join(Fn+"."+p[2:] for _,p in ga)}])',
             f'    (fun c ↦ by fin_cases c; exacts [{", ".join(Fn+"."+p[2:] for _,p in gb)}])',
             f'    (fun c ↦ by fin_cases c <;> rfl) {Fn}.{fm}',
             f'    (fun i cur next hb ↦ {SS}.sequential_generatedRoot_{mem}.sem hb)']
    L+=[f'  have ktot : Enc.A {tuple_of(ON,"Con",[f"w_{memo}_{k}" for k in range(6)])} =',
        f'      Enc.A {tuple_of(KE,"Ck",[f"w_{memk}_{k}" for k in range(6)])} := by',
        '    rw [toto, totk]',
        f'    exact Finset.sum_congr rfl fun j _ ↦ k{famo} _']
    return L


def header(name,doc,concl):
    return f'''set_option maxHeartbeats 400000000 in
/-- {doc} -/
theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : W_Stage_online_0.Params}}
    {{paramsK : W_Stage_keys_online_0_0.Params}}
    {{Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params}} {{oon : _}}
    (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {{Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK}} {{ok : _}}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck)
    (hq1 : 1 < W_Stage_online_0.modulus_0) (layout : RegularLayout W_Stage_online_0.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_0.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    {concl} := by
'''
def finish(colo,colk,C):
    L=[]
    for c in range(4):
        a,b=divmod(c,2)
        L.append(f'  have t{c} : {ON}.w_{colo}_{c} Con = {KE}.w_{colk}_{c} Ck :=')
        L.append(f'    row_eq_of_blk (C := {C}) (m := 25) fun cc ↦ congrFun (congrFun (kcol cc) {a}) {b}')
    L.append('  exact ⟨t0, t1, t2, t3⟩')
    return L
def concl(colo,colk):
    return ' ∧ '.join(f'{ON}.w_{colo}_{c} Con = {KE}.w_{colk}_{c} Ck' for c in range(4))
def prf_chain(spec):
    ch=Chain()
    body=[]
    (ca_o,ca_k,ca_N,ca_fo,ca_fk)=spec['cellsA']
    body+=ch.cells(ca_o,ca_k,ca_N,ca_fo,ca_fk,False)
    (ra_o,ra_k,M,ra_fo,ra_fk)=spec['rowA']
    body+=ch.rowsum(ra_o,ra_k,M,ra_fo,ra_fk,ca_o,ca_k)
    body+=ch.family_eqs(ra_o,ra_k,M)
    (cb_o,cb_k,cb_N,cb_fo,cb_fk)=spec['cellsB']
    body+=ch.cells(cb_o,cb_k,cb_N,cb_fo,cb_fk,True)
    (rb_o,rb_k,M,rb_fo,rb_fk)=spec['rowB']
    body+=ch.rowsum(rb_o,rb_k,M,rb_fo,rb_fk,cb_o,cb_k)
    body+=ch.family_eqs(rb_o,rb_k,M)
    return ch,body

def fresh_lemma():
    spec=dict(cellsA=(1181,1076,14400,'h1181_5','h1076_5'), rowA=(1182,1077,7200,'h1182_5','h1077_5'),
              cellsB=(1185,1080,14400,'h1185_5','h1080_5'), rowB=(1186,1081,7200,'h1186_5','h1081_5'))
    ch,body=prf_chain(spec)
    body+=ch.lut(1187,1082,7200,'h1187_5','h1082_5','sequential_parallel_generatedRoot_1187_18','sequential_parallel_generatedRoot_1082_18')
    body+=ch.family_eqs(1187,1082,7200)
    body+=ch.placement(1188,1083,7200,'h1188_5','h1083_5',1187)
    body+=column(ch,None,None,1188,1083,1207,1102,1219,1114,900,7200,8,'c46.h1207_5','c42.h1102_5','c46.h1219_5','c43.h1114_5')
    body+=finish(1219,1114,900)
    return header('fresh_total_agree0','**The fresh column totals of step 0 have the same keys** in the online stage (`w_1219`) and the keys stage (`w_1114`): every bundle of the chain (static and dynamic PRF cells, row sums, readout LUT, placement, member and column totals) computes its keys from the hashed root keys, the shared tables and switching ciphertexts and the previous keys alone.',concl(1219,1114))+'\n'.join(ch.A.lines+body)

def smudge():
    spec=dict(cellsA=(1235,1130,72000,'h1235_5','h1130_5'), rowA=(1236,1131,36000,'h1236_5','h1131_5'),
              cellsB=(1239,1134,72000,'h1239_5','h1134_5'), rowB=(1240,1135,36000,'h1240_5','h1135_5'))
    ch,body=prf_chain(spec)
    body+=ch.lut(1241,1136,36000,'h1241_5','h1136_5','sequential_parallel_generatedRoot_1241_18','sequential_parallel_generatedRoot_1136_18')
    body+=ch.family_eqs(1241,1136,36000)
    body+=ch.placement(1242,1137,36000,'h1242_5','h1137_5',1241)
    body+=column(ch,None,None,1242,1137,1261,1156,1273,1168,4500,36000,8,'c47.h1261_5','c44.h1156_5','c48.h1273_5','c44.h1168_5')
    body+=finish(1273,1168,4500)
    return header('smudge_total_agree0','**The smudge column totals of step 0 have the same keys** in the online stage (`w_1273`) and the keys stage (`w_1168`) (the chain of `fresh_total_agree0` with `36000` members).',concl(1273,1168))+'\n'.join(ch.A.lines+body)
def sigma():
    spec=dict(cellsA=(29,27,16,'h29_5','h27_5'), rowA=(30,28,8,'h30_5','h28_5'),
              cellsB=(33,31,16,'h33_5','h31_5'), rowB=(34,32,8,'h34_5','h32_5'))
    ch,body=prf_chain(spec)
    body+=ch.lut(35,33,8,'h35_5','h33_5','sequential_parallel_generatedRoot_35_18','sequential_parallel_generatedRoot_33_18')
    body+=ch.family_eqs(35,33,8)
    body+=ch.lut(36,34,8,'h36_5','h34_5','sequential_parallel_generatedRoot_36_18','sequential_parallel_generatedRoot_34_18')
    body+=monomial(ch,37,35,8,'h37_5','h35_5',36)
    body+=member_total(ch,37,35,56,54,8,'c0.h56_5','c0.h54_5')
    for c in range(4):
        a,b=divmod(c,2)
        body.append(f'  have t{c} : {ON}.w_56_{c} Con = {KE}.w_54_{c} Ck := congrFun (congrFun ktot {a}) {b}')
    body.append('  exact ⟨t0, t1, t2, t3⟩')
    return header('sigma_total_agree0','**The σ sampler totals of step 0 have the same keys** in the online stage (`w_56`) and the keys stage (`w_54`): static and dynamic PRF cells, row sums, the readout and sampling LUTs, the monomial placement and the member total.',concl(56,54))+'\n'.join(ch.A.lines+body)

# --------------------------------------------------------------------------- the state LUT loop
def lut_loop_lemma(loop_o, loop_k, field_o, field_k, sem_o, sem_k, A_o, A_k, fam_o, fam_k, seed_o, seed_k, sigmas, pre, given):
    A=Agree(links)
    A.given.update(given)
    aeq=[A.eq(a,b) for a,b in zip(A_o,A_k)]
    # init tuples
    _,ao=app_split(A.Fo[field_o][1]); _,ak=app_split(A.Fk[field_k][1])
    io=[strip_parens(x) for x in tuple_items(ao[2])]; ik=[strip_parens(x) for x in tuple_items(ak[2])]
    ieq=[A.eq(a,b) for a,b in zip(io,ik)]
    feq=[links[(a,b)] for a,b in zip(fam_o,fam_k)]
    seq=links[(seed_o,seed_k)]
    O=lambda w: f'{ON}.{w} Con'
    K=lambda w: f'{KE}.{w} Ck'
    M='ExactMatrix W_Stage_online_0.modulus_0 8 1 25'
    S=' × '.join([M]*10)+' × Unit'
    sel=f'MxxIR.selectInt (Int.ofNat i) {sigmas}'
    concl=' ∧ '.join(f'{O(f"w_{loop_o}_{c}")} = {K(f"w_{loop_k}_{c}")}' for c in range(2,10))
    body=list(A.lines)
    h0line=f'  have h0 : p ({", (".join(O(x) for x in io)}{", ()" + ")"*len(io)} = p ({", (".join(K(x) for x in ik)}{", ()" + ")"*len(ik)} := by\n    rw [{", ".join(dict.fromkeys(ieq))}]'
    body+=[f'  let p : ({S}) → (Fin 2 → Fin 2 → {M}) × (Fin 2 → Fin 2 → {M}) := fun s ↦',
      '    (![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]],',
      '      ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]])',
      f'  let fam : Fin 2 → Fin 2 → Fin 4 → {M} := ![![{O(fam_o[0])}, {O(fam_o[1])}], ![{O(fam_o[2])}, {O(fam_o[3])}]]',
      f'  let famN : Nat → Fin 2 → Fin 2 → {M} := fun j k l ↦ if h : j < 4 then fam k l ⟨j, h⟩ else 0',
      f'  let Ak : Fin 2 → Fin 2 → {M} := ![![{O(A_o[0])}, {O(A_o[2])}], ![{O(A_o[1])}, {O(A_o[3])}]]',
      f'  let F : Nat → (Fin 2 → Fin 2 → {M}) × (Fin 2 → Fin 2 → {M}) → (Fin 2 → Fin 2 → {M}) × (Fin 2 → Fin 2 → {M}) :=',
      f'    fun i x ↦ (x.1 + branchKeyF backend 4096 25 ({sel}).toNat Ak',
      f'      (keyPairG hashModel ({O(seed_o)}) W_Stage_online_0.modulus_0 25 {pre} [.integer ({sel})]) x.2, famN (i + 1))',
      h0line,
      '  have hl := iter_link_gen p p F (count := 3)',
      '    (fun i cur nxt hi hb ↦ by',
      f'      obtain ⟨hk, hsw⟩ := LutRoundStep.keys_1 (by norm_num) hq1 layout hlayout hexact ({sem_o}.sem hb)',
      '      obtain ⟨n0, n1, n2, n3⟩ := hsw (by omega)',
      '      refine Prod.ext hk ?_',
      '      funext k l',
      '      fin_cases k <;> fin_cases l',
      '      · show _ = famN (i + 1) 0 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n0',
      '      · show _ = famN (i + 1) 0 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n1',
      '      · show _ = famN (i + 1) 1 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n2',
      '      · show _ = famN (i + 1) 1 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n3)',
      '    (fun i cur nxt hi hb ↦ by',
      f'      obtain ⟨hk, hsw⟩ := LutRoundStep.keys_1 (by norm_num) hq1 layout hlayout hexact ({sem_k}.sem hb)',
      '      obtain ⟨n0, n1, n2, n3⟩ := hsw (by omega)',
      f'      rw [← {aeq[0]}, ← {aeq[1]}, ← {aeq[2]}, ← {aeq[3]}, ← {seq}] at hk',
      f'      rw [← {feq[0]}] at n0',
      f'      rw [← {feq[1]}] at n1',
      f'      rw [← {feq[2]}] at n2',
      f'      rw [← {feq[3]}] at n3',
      '      refine Prod.ext hk ?_',
      '      funext k l',
      '      fin_cases k <;> fin_cases l',
      '      · show _ = famN (i + 1) 0 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n0',
      '      · show _ = famN (i + 1) 0 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n1',
      '      · show _ = famN (i + 1) 1 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n2',
      '      · show _ = famN (i + 1) 1 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n3)',
      f'    Fon.{A.Fo[field_o][0][2:]} Fk.{A.Fk[field_k][0][2:]} h0']
    names=[]
    for c in range(2,10):
        part='1' if c<6 else '2'; idx=(c-2)%4; a,b=divmod(idx,2)
        body.append(f'  have t{c} : {O(f"w_{loop_o}_{c}")} = {K(f"w_{loop_k}_{c}")} := congrFun (congrFun (congrArg Prod.{"fst" if part=="1" else "snd"} hl) {a}) {b}')
        names.append(f't{c}')
    body.append(f'  exact ⟨{", ".join(names)}⟩')
    return header(f'lut_loop_agree0_{loop_o}', f'**The root LUT loop `sequential_generatedRoot_{loop_o}` of step 0 has the same keys and carried switching ciphertexts** as the keys stage\'s `sequential_generatedRoot_{loop_k}` (`iter_link_gen` with the step forms of `LutRoundStep.keys_1`).', concl)+'\n'.join(body)

# --------------------------------------------------------------------------- the cₓ targets
O=lambda w: f'({ON}.{w} Con)'
K=lambda w: f'({KE}.{w} Ck)'
RO='MxxExponentLut.rowOf'

class CX:
    """Match an online key expression (`mul`, `add`, `rowstack` = rows `1, 2` of a stacked
    product, `zadd` = sum with a zero-count loop's zero, `wire`, `raw`) against a keys-stage wire."""
    def __init__(self, given):
        self.A=Agree(links); self.A.given.update(given)
        self.n=0; self.lines=[]
    def fresh(self): self.n+=1; return f'x{self.n}'
    def text(self,e):
        k=e[0]
        if k=='wire': return O(e[1])
        if k=='mul': return f'({self.text(e[1])} * {self.text(e[2])})'
        if k=='add': return f'({self.text(e[1])} + {self.text(e[2])})'
        if k=='rowstack': return f'{RO} (matrixMul {O(e[1])} {O(e[2])}) {e[3]}'
        if k=='zadd': return self.text(e[1])
        if k=='raw': return e[1]
        raise Exception(k)
    def let(self,w):
        return self.A.Lk.get(w,'')
    def match(self,e,wk):
        """Return the name of a proof of `text(e) = keys.wk`."""
        k=e[0]
        name=self.fresh()
        lhs='('+self.text(e)+' : ExactMatrix W_Stage_online_0.modulus_0 8 1 25)'; rhs=K(wk)
        if k=='wire':
            return self.A.eq(e[1],wk)
        if k=='raw':
            return e[2]
        L=self.let(wk)
        if k in ('mul','add'):
            op='matrixMul' if k=='mul' else 'matrixAdd'
            m=re.match(r'^MxxRuntime\.'+op+r' (\S+) (\S+)$',L)
            if not m: raise Exception(f'{k} vs {wk}: {L}')
            a=self.match(e[1],m.group(1)); b=self.match(e[2],m.group(2))
            sym='*' if k=='mul' else '+'
            self.lines.append(f'  have {name} : {lhs} = {rhs} := congrArg₂ (· {sym} ·) {a} {b}')
            return name
        if k=='zadd':
            m=re.match(r'^MxxRuntime\.matrixAdd (\S+) (\S+)$',L)
            if not m: raise Exception(f'zadd vs {wk}: {L}')
            z=m.group(1)
            lo=self.A.loop_k.get(z)
            if not lo or lo[1]!='Int.toNat (0)': raise Exception(f'zadd zero {z}')
            p,cnt,idx,init=lo
            proj='s'+'.2'*idx+'.1'
            b=self.match(e[1],m.group(2))
            self.lines.append(f'  have {name} : {lhs} = {rhs} := by')
            self.lines.append(f'    have hz : {K(z)} = 0 := (congrArg (fun s ↦ {proj}) (iterRuns_zero_eq Fk.{p[2:]} rfl)).trans rfl')
            self.lines.append(f'    show _ = matrixAdd {K(z)} {K(m.group(2))}')
            self.lines.append(f'    rw [hz]')
            self.lines.append(f'    exact {b}.trans (zero_add _).symm')
            return name
        if k=='rowstack':
            So,Do,r=e[1],e[2],e[3]
            pk=self.A.prod_k.get(wk)
            if not pk or pk[0]!='sliceMatrix': raise Exception(f'rowstack vs {wk}')
            ak=[strip_parens(x) for x in pk[2]]
            if ak[1:5]!=[str(r),str(r+1),'0','25']: raise Exception(f'slice rows {ak}')
            mk=re.match(r'^MxxRuntime\.matrixMul (\S+) (\S+)$',self.A.Lk.get(ak[0],''))
            so=self.A.stack(self.A.Fo,self.A.prod_o,So); sk=self.A.stack(self.A.Fk,self.A.prod_k,mk.group(1))
            es=[self.A.eq(so[x],sk[x]) for x in ('l1','l2','r1','r2')]
            ed=self.A.eq(Do,mk.group(2))
            args=' '.join('Fon.'+so['p'+x][2:] for x in ('l01','l','r01','r','both'))+' '+' '.join('Fk.'+sk['p'+x][2:] for x in ('l01','l','r01','r','both'))
            self.lines.append(f'  have {name} : {lhs} = {rhs} := by')
            self.lines.append(f'    rw [slice_rowOfG (by norm_num) Fk.{pk[1][2:]}]')
            self.lines.append(f'    exact (stack_rows_agree {args} {" ".join(es)} {ed}).{r}')
            return name
        raise Exception(k)

PRIV='[114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]'
LABELS=[('w_382_0','w_387_0','w_399_0','w_403_0','w_429_0','w_409_0','w_418_0','w_434_0'),
        ('w_475_0','w_495_0','w_511_0','w_513_0','w_544_0','w_525_0','w_535_0','w_546_0'),
        ('w_587_0','w_592_0','w_604_0','w_606_0','w_627_0','w_612_0','w_619_0','w_629_0'),
        ('w_642_0','w_647_0','w_658_0','w_660_0','w_681_0','w_666_0','w_673_0','w_683_0'),
        ('w_696_0','w_701_0','w_712_0','w_714_0','w_735_0','w_720_0','w_727_0','w_737_0')]
AUTS=[('w_776_0','w_794_0','w_804_0','w_810_0'),('w_834_0','w_852_0','w_862_0','w_868_0'),
      ('w_892_0','w_910_0','w_920_0','w_926_0'),('w_950_0','w_968_0','w_978_0','w_984_0'),
      ('w_1008_0','w_1026_0','w_1036_0','w_1042_0'),('w_1066_0','w_1084_0','w_1094_0','w_1100_0'),
      ('w_1124_0','w_1142_0','w_1152_0','w_1158_0')]
HASHPRIV={(0,0):('w_17_0','c0.h17_0'),(0,1):('w_21_0','c0.h21_0'),(1,0):('w_18_0','c0.h18_0'),(1,1):('w_22_0','c0.h22_0')}

def build():
    given={('w_56_0','w_54_0'):'s0',('w_56_1','w_54_1'):'s1',('w_56_2','w_54_2'):'s2',('w_56_3','w_54_3'):'s3'}
    for c in range(2,10): given[(f'w_317_{c}',f'w_315_{c}')]=f'l{c}'
    cx=CX(given)
    l0=packlist(cx.A.Fk,'h1057_0'); l1=packlist(cx.A.Fk,'h1266_0')
    pre=[]  # raw proofs
    # raw: one-public label keys and one-private root keys
    raw={}
    for k in range(2):
        nm=f'op{k}'
        hw,hp=('w_62_0','c1.h62_0') if k==0 else ('w_407_0','c14.h407_0')
        txt=f'labelKeysG (n := 8) hashModel ({ON}.w_7_0 Con) W_Stage_online_0.modulus_0 25 labelPre 1 {k} 1'
        pre.append(f'  have {nm} : {txt} = {K(hw)} := by')
        pre.append(f'    rw [L.e7]; exact (keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) {k} 1 (by norm_num) Fk.{hp}).symm')
        raw[('op',k)]=('raw',txt,nm)
        for j in range(2):
            nm=f'pk{k}{j}'; hw,hp=HASHPRIV[(k,j)]
            txt=f'rootKey (n := 8) hashModel ({ON}.w_7_0 Con) {PRIV} [] W_Stage_online_0.modulus_0 25 {k} {j}'
            pre.append(f'  have {nm} : {txt} = {K(hw)} := by')
            pre.append(f'    rw [L.e7, hashSample_eq Fk.{hp}]; rfl')
            raw[('pk',k,j)]=('raw',txt,nm)
    W=lambda w:('wire',w)
    members=[]   # (defn text as function of k, [ast row0, ast row1])
    # PRIV
    for j in range(2):
        members.append((f'fun k ↦ rootKey (n := 8) hashModel ({ON}.w_7_0 Con) {PRIV} [] W_Stage_online_0.modulus_0 25 k {j} * {O("w_236_0")}',
            [('mul',raw[('pk',k,j)],W('w_236_0')) for k in range(2)]))
    # PUB
    members.append((f'fun k ↦ sigmaKey0 Con k * {O("w_243_0")}', [('mul',('mul',W('w_56_1'),W('w_62_0')),W('w_243_0')),('mul',('mul',W('w_56_3'),W('w_62_0')),W('w_243_0'))]))
    members.append((f'fun k ↦ onePubKey0 Con k * {O("w_243_0")}', [('mul',('mul',raw[('op',k)],W('w_69_0')),W('w_243_0')) for k in range(2)]))
    # STATE
    for j,(D,) in enumerate([('w_345_0',),('w_365_0',)]):
        a=[f'w_317_{2+j}',f'w_317_{4+j}']
        members.append((f'fun k ↦ ![{O(a[0])} + {RO} (matrixMul {O("w_327_0")} {O(D)}) 1, {O(a[1])} + {RO} (matrixMul {O("w_327_0")} {O(D)}) 2] k * {O("w_355_0")}',
            [('mul',('add',W(a[k]),('rowstack','w_327_0',D,k+1)),W('w_355_0')) for k in range(2)]))
    # LABELS
    for (P,D1,Dg,Q0,R0,D2,Q1,R1) in LABELS:
        for j,(Q,R) in enumerate([(Q0,Q1),(R0,R1)]):
            members.append((f'fun k ↦ {RO} (matrixMul {O(P)} {O(D1)}) k.succ * {O(Dg)} * {O(Q)} + {RO} (matrixMul {O(P)} {O(D2)}) k.succ * {O(Dg)} * {O(R)}',
                [('add',('mul',('mul',('rowstack',P,D1,k+1),W(Dg)),W(Q)),('mul',('mul',('rowstack',P,D2,k+1),W(Dg)),W(R))) for k in range(2)]))
    # SIGMA target
    members.append((f'fun k ↦ ![{O("w_56_1")}, {O("w_56_3")}] k * {O("w_746_0")} * {O("w_752_0")}',
        [('mul',('mul',W(f'w_56_{1+2*k}'),W('w_746_0')),W('w_752_0')) for k in range(2)]))
    members.append((f'fun k ↦ onePubKey0 Con k * {O("w_752_0")}', [('mul',('mul',raw[('op',k)],W('w_69_0')),W('w_752_0')) for k in range(2)]))
    # AUT
    for (S,D,Dr,Dg) in AUTS:
        members.append((f'fun k ↦ {RO} (matrixMul {O(S)} {O(D)}) k.succ * {O(Dr)} * {O(Dg)}',
            [('mul',('mul',('zadd',('rowstack',S,D,k+1)),W(Dr)),W(Dg)) for k in range(2)]))
        members.append((f'fun k ↦ onePubKey0 Con k * {O(Dg)}', [('mul',('mul',raw[('op',k)],W('w_69_0')),W(Dg)) for k in range(2)]))
    # zeros
    for _ in range(4):
        members.append(('fun _ ↦ 0', None))
    assert len(members)==36
    proofs=[]
    for m,(txt,asts) in enumerate(members):
        if asts is None: proofs.append(None); continue
        ps=[]
        for k in range(2):
            wk=(l0 if k==0 else l1)[m]
            try: ps.append(cx.match(asts[k],wk))
            except Exception as e:
                raise Exception(f'cx member {m} row {k} ({wk}): {e}')
        proofs.append(ps)
    return cx,pre,members,proofs,l0,l1

def cx_lemma():
    cx,pre,members,proofs,l0,l1=build()
    out=[]
    out.append('set_option maxHeartbeats 16000000 in')
    out.append('/-- The online `cₓ` keys of step 0, member by member of the 36-family `w_1162`, as the online')
    out.append('target theorems state them (`online0_target_priv`, `online0_target_pub` with `sigmaKey0` and')
    out.append('`onePubKey0`, `online0_state_lut`/`online0_target_state`, `online0_label_*`, `online0_target_sigma`,')
    out.append('`online0_aut_*`, and the four zero rows): member `2 t + j` has the keys `K_t · j`. -/')
    out.append(f'noncomputable def cxKeys0 {{backend : BackendContext}} {{hashModel : HashModel}}')
    out.append(f'    {{params : W_Stage_online_0.Params}} (Con : {ON}.Ctx backend hashModel params) :')
    out.append(f'    Fin 36 → Fin 2 → ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=')
    out.append('  ![' + ',\n    '.join('(' + t.replace('↦ ', '↦ (', 1) + ' : ExactMatrix W_Stage_online_0.modulus_0 8 1 25))' for t,_ in members) + ']')
    out.append('')
    concl='∀ j : Fin 36, cxKeys0 Con j 0 = W_Stage_keys_online_0_0.generatedRoot.w_1057_0 Ck j ∧\n      cxKeys0 Con j 1 = W_Stage_keys_online_0_0.generatedRoot.w_1266_0 Ck j'
    out.append(header('cx_keys_agree0','**The online `cₓ` keys of step 0 are the keys stage\'s two `cₓ` rows** (`w_1057`, `w_1266`), member by member: every target\'s keys are computed by both stages from the same hashed root keys, shared artifacts and the agreeing σ sampler and state LUT keys (`sigma_total_agree0`, `lut_loop_agree0_317`).',concl))
    out.append('  obtain ⟨s0, s1, s2, s3⟩ := sigma_total_agree0 Fon Fk L hq1 layout hlayout hexact')
    out.append('  obtain ⟨l2, l3, l4, l5, l6, l7, l8, l9⟩ := lut_loop_agree0_317 Fon Fk L hq1 layout hlayout hexact')
    out+=pre
    out+=cx.A.lines
    out+=cx.lines
    out.append('  intro j')
    out.append('  fin_cases j')
    for m,ps in enumerate(proofs):
        if ps is None:
            out.append(f'  · obtain ⟨_, e0⟩ := familyPack_get Fk.c41.h1057_0 ⟨{m}, by norm_num⟩')
            out.append(f'    obtain ⟨_, e1⟩ := familyPack_get Fk.c49.h1266_0 ⟨{m}, by norm_num⟩')
            z=f'(rfl : (0 : ExactMatrix W_Stage_online_0.modulus_0 8 1 25) = {K(l0[m])})'
            z1=f'(rfl : (0 : ExactMatrix W_Stage_online_0.modulus_0 8 1 25) = {K(l1[m])})'
            out.append(f'    exact ⟨{z}.trans e0.symm, {z1}.trans e1.symm⟩')
            continue
        out.append(f'  · obtain ⟨_, e0⟩ := familyPack_get Fk.c41.h1057_0 ⟨{m}, by norm_num⟩')
        out.append(f'    obtain ⟨_, e1⟩ := familyPack_get Fk.c49.h1266_0 ⟨{m}, by norm_num⟩')
        out.append(f'    exact ⟨{ps[0]}.trans e0.symm, {ps[1]}.trans e1.symm⟩')
    return '\n'.join(out)

# --------------------------------------------------------------------------- the public family
def fam_lemma(name, doc, fo, fk, fieldo, fieldk):
    A=Agree(links)
    lo=packlist(A.Fo,fieldo); lk=packlist(A.Fk,fieldk)
    assert len(lo)==len(lk)
    names=[A.eq(a,b) for a,b in zip(lo,lk)]
    po=A.Fo[fieldo][0][2:]; pk=A.Fk[fieldk][0][2:]
    body='\n'.join(A.lines)
    return f'''set_option maxHeartbeats 64000000 in
/-- {doc} -/
theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : W_Stage_online_0.Params}}
    {{paramsK : W_Stage_keys_online_0_0.Params}}
    {{Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params}} {{oon : _}}
    (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {{Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK}} {{ok : _}}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) (L : Link0 Con Ck) :
    {ON}.{fo} Con = {KE}.{fk} Ck := by
{body}
  have hp := Fon.{po}
  rw [{", ".join(dict.fromkeys(names))}] at hp
  exact familyPack_eq hp Fk.{pk} rfl
'''

# --------------------------------------------------------------------------- block-diagonal digits

def diag_lemmas():
    def chain(S, base, fields):
        out = [f"  have e0 := isDiag_self ({S}.{base} c)"]
        for k, f in enumerate(fields, 1):
            out.append(f"  have e{k} := isDiag_step (k := {k}) (by norm_num) (by norm_num) (by norm_num) (by norm_num) e{k - 1} F.{f}")
        out.append(f"  exact e{len(fields)}")
        return '\n'.join(out)
    K = 'W_Stage_keys_online_0_0.parallel_generatedRoot_1267'
    O = 'W_Stage_online_0.parallel_generatedRoot_1281'

    def lem(name, doc, S, base, out, fields):
        assert len(fields) == 24
        return f"""/-- {doc} -/
theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : {S.split('.')[0]}.Params}}
    {{i : Nat}} {{c : {S}.Ctx backend hashModel params i}} {{o : _}}
    (F : {S}.Facts c o) : IsDiag ({S}.{base} c) ({S}.{out} c) := by
{chain(S, base, fields)}
"""
    kf0 = ['c0.h27_0'] + [f'c0.h27_0_{k}' for k in range(1, 14)] + [f'c1.h27_0_{k}' for k in range(14, 23)] + ['c1.h28_0']
    kf1 = ['c3.h27_0_23'] + [f'c3.h27_0_{k}' for k in range(24, 42)] + [f'c4.h27_0_{k}' for k in range(42, 46)] + ['c4.h80_0']
    ks0 = ['c2.h45_0'] + [f'c2.h45_0_{k}' for k in range(1, 23)] + ['c2.h46_0']
    ks1 = ['c4.h45_0_23'] + [f'c4.h45_0_{k}' for k in range(24, 28)] + [f'c5.h45_0_{k}' for k in range(28, 46)] + ['c5.h97_0']
    of0 = ['c0.h31_0'] + [f'c0.h31_0_{k}' for k in range(1, 13)] + [f'c1.h31_0_{k}' for k in range(13, 23)] + ['c1.h32_0']
    os0 = ['c2.h49_0'] + [f'c2.h49_0_{k}' for k in range(1, 23)] + ['c2.h50_0']
    return '\n'.join([
        lem('keys1267_diagF0', 'The keys stage, row `0`: the fresh digits act on `diag(G, …, G)`.', K, 'w_27_0', 'w_28_0', kf0),
        lem('keys1267_diagF1', 'The keys stage, row `1`: the fresh digits act on `diag(G, …, G)`.', K, 'w_27_0', 'w_80_0', kf1),
        lem('keys1267_diagS0', 'The keys stage, row `0`: the smudge digits act on `diag(h, …, h)`.', K, 'w_45_0', 'w_46_0', ks0),
        lem('keys1267_diagS1', 'The keys stage, row `1`: the smudge digits act on `diag(h, …, h)`.', K, 'w_45_0', 'w_97_0', ks1),
        lem('online1281_diagF0', 'The online part, prime `0`: the fresh digits act on `diag(G, …, G)`.', O, 'w_31_0', 'w_32_0', of0),
        lem('online1281_diagS0', 'The online part, prime `0`: the smudge digits act on `diag(h, …, h)`.', O, 'w_49_0', 'w_50_0', os0),
    ]) + '\n'


def gadget_lemma():
    A = Agree(links)
    a = A.eq('w_57_0', 'w_55_0'); b = A.eq('w_62_0', 'w_60_0'); c = A.eq('w_69_0', 'w_67_0')
    return f"""set_option maxHeartbeats 16000000 in
/-- The gadget of step 0 and its digits (`G⁻¹(G)` for the σ carrier and the one-public row) agree
between the stages. -/
theorem gadget_agree0 {{backend : BackendContext}} {{hashModel : HashModel}} {{params : W_Stage_online_0.Params}}
    {{paramsK : W_Stage_keys_online_0_0.Params}}
    {{Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params}} {{oon : _}}
    (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {{Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK}} {{ok : _}}
    (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok) :
    {ON}.w_57_0 Con = {KE}.w_55_0 Ck ∧ {ON}.w_62_0 Con = {KE}.w_60_0 Ck ∧
      {ON}.w_69_0 Con = {KE}.w_67_0 Ck := by
""" + '\n'.join(A.lines) + f'\n  exact ⟨{a}, {b}, {c}⟩\n'


TEMPLATE = r"""import Dio2Preimage
import Dio2Initial
import Dio2Online0PartPrimes
import Dio2Online0PartFresh
import Dio2KeysLoops
import Dio2Online0Fresh
import Dio2Online0Refresh
import Dio2Online0Helper
import Dio2Online0State
import Dio2Online0Label
import Gen_W_Stage_keys_online_0_0_Scopes
import W_Stage_keys_online_0_0_Facts

/-!
# Key agreement for the refresh parts of step 0 (prime 0)

Generated by `crates/io/lean/dio2/generator/gen_key_agree0.py`.

`online0_part_prime0` (`Dio2Online0PartPrimes`) needs `hpre`: the preimage `P` of part `i`
satisfies `B_k P = (K_g,k D_top + K₁,k D_bot - K_x,k Y) + K_f,k + K_s,k` for both mask components
`k`, with the keys `K_*` of the online rows. The preimage solves `B_0 P = K` for member `i` of the
keys stage's family `dio2/key/0` (`claim_preimages`). The keys stage (`W_Stage_keys_online_0_0`,
root `parallel_generatedRoot_1267`) runs the same graph on the key rows instead of the ciphertext
rows and with its own label payloads, so no row or payload of the two stages is assumed equal:
every key the combination needs is shown to be computed by both stages from the same hashed root
keys, shared tables, switching ciphertexts and digits (the role key lemmas of `Dio2Keys`,
`Dio2KeysLoops`, refresh_top's `GenKey`, and the determinism of decompositions, rebases and
hashes).

* `keys_part_rows0`: rows `0` and `1` of the keys stage's prime-0 output of part `i` are
  `[g | c₁] D - cₓ Y + fresh + smudge` over its row-`k` inputs; the keys stage's two column slice
  loops per root (one per row, ten over the five prime roots) are read by `slice_column_total`.
* `part_agree0`: `hpre` for one part from the agreement of its inputs (the public matrix `A'`,
  the gadget, the smudge constant) and of the key rows; `D`, `Y` and the fresh and smudge digits
  agree by determinism.
* `Link0`, `claim_link0`: the inputs both stages read from the same `W_Claim` source; the
  preimages solve against the anchor `B_0` and the keys stage's family.
* `pub_family_agree0`, `gadget_agree0`, `fresh_total_agree0`, `sigma_total_agree0`,
  `smudge_total_agree0`, `lut_loop_agree0_317`, `cx_keys_agree0`: the key agreements (public
  family, gadget digits, the fresh, σ and smudge PRF chains, the state LUT loop, and the 36 `cₓ`
  targets), quantified over members, never unrolled over the PRF members or the loop iterations.
* `online0_hpre0_full`, `claim_hpre0`: `hpre` with the online keys `sigmaKey0`, `onePubKey0`,
  `cxKeys0`, `freshKey0 … freshCells0`, `smudgeKey0 … smudgeCells0`; `claim_anchor0`,
  `claim_hint0`: the hint's keys `Bk` are the rows of the same anchor `B_0`.
-/

set_option maxHeartbeats 32000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenFinal GenDecode

variable {n : Nat}

/-! ## Generic lemmas -/

/-- Row `k` of a solved preimage equation. -/
theorem rowOf_mul_of_solved {q rows inner cols : Nat} {A : ExactMatrix q n rows inner}
    {T : ExactMatrix q n rows cols} {P : ExactMatrix q n inner cols} (h : A * P = T)
    (k : Fin rows) : MxxExponentLut.rowOf A k * P = MxxExponentLut.rowOf T k := by
  rw [← rowOf_mul, h]

/-- A block-diagonal matrix is determined by its block. -/
theorem isDiag_unique {q R C a b : Nat} (ha : 0 < a) (hb : 0 < b) {x : ExactMatrix q n a b}
    {Y Y' : ExactMatrix q n R C} (hY : IsDiag x Y) (hY' : IsDiag x Y') : Y = Y' := by
  funext i c
  rw [hY i c ⟨i.val % a, Nat.mod_lt _ ha⟩ ⟨c.val % b, Nat.mod_lt _ hb⟩ rfl rfl,
    hY' i c ⟨i.val % a, Nat.mod_lt _ ha⟩ ⟨c.val % b, Nat.mod_lt _ hb⟩ rfl rfl]


/-- Block `c` of the key row of a row of cells is the key of cell `c`. -/
theorem blk_cellsKey {q C m : Nat} (K : Fin C → Fin 2 → ExactMatrix q n 1 m) (k : Fin 2)
    (c : Fin C) : blk (cellsKey K k) c = K c k := by
  funext r l
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; exact l.elim0)
  have hd : (c.val * m + l.val) / m = c.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt l.isLt, zero_add]
  have hmd : (c.val * m + l.val) % m = l.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt l.isLt]
  simp only [blk, cellsKey, divFin, modFin]
  congr 2

/-- **Two stacked products agree on their key rows.** Rows `1` and `2` of `[c; A₀; A₁] d` depend
on the key rows `A` and `d` alone, not on the ciphertext row `c`. -/
theorem stack_rows_agree {q m m' : Nat} {l0 l1 l2 r0 r1 r2 l0' l1' l2' r0' r1' r2' : ExactMatrix q n 1 m}
    {l01 r01 l01' r01' : ExactMatrix q n (1 + 1) m} {l r l' r' : ExactMatrix q n (1 + 1 + 1) m}
    {both both' : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows l0 l1 l01) (hl : concatRows l01 l2 l)
    (hr01 : concatRows r0 r1 r01) (hr : concatRows r01 r2 r) (hboth : concatColumns l r both)
    (hl01' : concatRows l0' l1' l01') (hl' : concatRows l01' l2' l')
    (hr01' : concatRows r0' r1' r01') (hr' : concatRows r01' r2' r') (hboth' : concatColumns l' r' both')
    (e1 : l1 = l1') (e2 : l2 = l2') (e3 : r1 = r1') (e4 : r2 = r2') {d d' : ExactMatrix q n (m + m) m'}
    (ed : d = d') :
    MxxExponentLut.rowOf (both * d) 1 = MxxExponentLut.rowOf (both' * d') 1 ∧
      MxxExponentLut.rowOf (both * d) 2 = MxxExponentLut.rowOf (both' * d') 2 := by
  obtain ⟨-, a1, a2⟩ := stacked_row hl01 hl hr01 hr hboth d
  obtain ⟨-, b1, b2⟩ := stacked_row hl01' hl' hr01' hr' hboth' d'
  rw [a1, a2, b1, b2, e1, e2, e3, e4, ed]
  exact ⟨rfl, rfl⟩


/-! ## The block-diagonal digits of a part -/

@@DIAG@@
/-! ## The keys stage's prime-0 output -/

section KeysRows

open W_Stage_keys_online_0_0

/-- **The two rows of the keys stage's prime-0 output of part `i`** (`parallel_generatedRoot_1267`):
row `k` is the part's combination over the keys stage's row-`k` inputs (`g`, `c₁`, `cₓ`, the fresh
and smudge column rows), `[g | c₁] D - cₓ Y + fresh + smudge`. -/
theorem keys_part_rows0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (d : parallel_generatedRoot_1267.Ctx backend hashModel params i) {o : _}
    (F : parallel_generatedRoot_1267.Facts d o) :
    MxxExponentLut.rowOf o 0 = ((parallel_generatedRoot_1267.w_0_0 d * topRows (parallel_generatedRoot_1267.w_7_0 d) +
        parallel_generatedRoot_1267.w_1_0 d * bottomRows (parallel_generatedRoot_1267.w_7_0 d) -
        parallel_generatedRoot_1267.w_9_0 d * parallel_generatedRoot_1267.w_21_0 d) +
        parallel_generatedRoot_1267.w_41_0 d) + parallel_generatedRoot_1267.w_52_0 d ∧
    MxxExponentLut.rowOf o 1 = ((parallel_generatedRoot_1267.w_54_0 d * topRows (parallel_generatedRoot_1267.w_60_0 d) +
        parallel_generatedRoot_1267.w_55_0 d * bottomRows (parallel_generatedRoot_1267.w_60_0 d) -
        parallel_generatedRoot_1267.w_62_0 d * parallel_generatedRoot_1267.w_74_0 d) +
        parallel_generatedRoot_1267.w_93_0 d) + parallel_generatedRoot_1267.w_103_0 d := by
  rw [F.c5.hout]
  obtain ⟨h0, h1⟩ := concatRows_two F.c5.h105_0
  refine ⟨?_, ?_⟩
  · show MxxExponentLut.rowOf (parallel_generatedRoot_1267.w_105_0 d) 0 = _
    rw [h0, ← concatColumns_mul F.c0.h2_0]
    rfl
  · show MxxExponentLut.rowOf (parallel_generatedRoot_1267.w_105_0 d) 1 = _
    rw [h1, ← concatColumns_mul F.c2.h56_0]
    rfl

end KeysRows

/-! ## One part, prime `0`: `hpre` from the agreement of the inputs -/

section PartAgree

/-- The online fresh-row keys of `online0_part_fresh0` (cells `25 i + c` of the cell keys `K`). -/
noncomputable def freshKey0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} {i : Nat}
    (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params i) (hi : i < 36)
    (K : Fin 900 → Fin 2 → ExactMatrix W_Stage_online_0.modulus_0 8 1 25) (k : Fin 2) :
    ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  fun _ l ↦ (blk (C := 25) (m := 25) (cellsKey (fun cc : Fin 25 ↦ K ⟨25 * i + cc.val, by omega⟩) k *
      (W_Stage_online_0.parallel_generatedRoot_1281.w_37_0 c)) (RefreshTopProof.GenFinal.colOf (C := 25) l) *
    (W_Stage_online_0.parallel_generatedRoot_1281.w_42_0 c)) 0 0

/-- The online smudge-row keys of `online0_part_smudge0` (cells `125 i + c` of the cell keys `K`). -/
noncomputable def smudgeKey0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} {i : Nat}
    (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params i) (hi : i < 36)
    (K : Fin 4500 → Fin 2 → ExactMatrix W_Stage_online_0.modulus_0 8 1 25) (k : Fin 2) :
    ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  cellsKey (fun cc : Fin 25 ↦ K ⟨125 * i + cc.val, by omega⟩) k *
    W_Stage_online_0.parallel_generatedRoot_1281.w_55_0 c

open W_Stage_online_0 in
set_option maxHeartbeats 64000000 in
/-- **`hpre` of `online0_part_prime0` for one part.** The preimage `P` (the part's input `w_1`)
solves `A P = o` for the keys stage's prime-0 output `o` of the same part (`hP`). If the keys
stage's part reads the same public matrix `A'` (`hA`), gadget (`hG`) and smudge constant (`hh`),
and its row-`k` inputs are the online keys (`hKg`, `hK₁`, `hKx`: the keys of `g`, `c₁`, `cₓ`;
`hK`, `hKs`: the fresh and smudge cell keys are the blocks of its fresh and smudge column rows),
then `rowOf A k P` is the online combination of keys, for both `k`. -/
theorem part_agree0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {params' : W_Stage_keys_online_0_0.Params} {i : Nat} (hi : i < 36)
    (c : parallel_generatedRoot_1281.Ctx backend hashModel params i) {oc : _}
    (Fc : parallel_generatedRoot_1281.Facts c oc)
    (d : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.Ctx backend hashModel params' i) {od : _}
    (Fd : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.Facts d od)
    (hq1 : 1 < modulus_0)
    (hA : parallel_generatedRoot_1281.w_8_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_4_0 d)
    (hG : parallel_generatedRoot_1281.w_31_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_27_0 d)
    (hh : parallel_generatedRoot_1281.w_49_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_45_0 d)
    {A : ExactMatrix modulus_0 8 2 54} (hP : A * parallel_generatedRoot_1281.w_1_0 c = od)
    {Kg K₁ Kx : Fin 2 → ExactMatrix modulus_0 8 1 25}
    (hKg : Kg 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_0_0 d ∧
      Kg 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_54_0 d)
    (hK₁ : K₁ 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_1_0 d ∧
      K₁ 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_55_0 d)
    (hKx : Kx 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_9_0 d ∧
      Kx 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_62_0 d)
    {K : Fin 900 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    (hK : ∀ cc, K cc 0 = blk (C := 900) (m := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_25_0 d) cc ∧
      K cc 1 = blk (C := 900) (m := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_78_0 d) cc)
    {Ksm : Fin 4500 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    (hKs : ∀ cc, Ksm cc 0 = blk (C := 4500) (m := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_43_0 d) cc ∧
      Ksm cc 1 = blk (C := 4500) (m := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_95_0 d) cc) :
    ∀ k', MxxExponentLut.rowOf A k' * parallel_generatedRoot_1281.w_1_0 c =
      (Kg k' * topRows (parallel_generatedRoot_1281.w_11_0 c) + K₁ k' * bottomRows (parallel_generatedRoot_1281.w_11_0 c) -
        Kx k' * parallel_generatedRoot_1281.w_25_0 c) + freshKey0 c hi K k' + smudgeKey0 c hi Ksm k' := by
  have hn : (0 : Nat) < 8 := by norm_num
  obtain ⟨r0, r1⟩ := keys_part_rows0 d Fd
  -- `D`: the digits of `Q_q A'`.
  have e9 : parallel_generatedRoot_1281.w_9_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_5_0 d := by
    show matrixMulScalarLeft _ (parallel_generatedRoot_1281.w_8_0 c) = matrixMulScalarLeft _ _
    rw [hA]; rfl
  have e9' : parallel_generatedRoot_1281.w_9_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_58_0 d := by
    show matrixMulScalarLeft _ (parallel_generatedRoot_1281.w_8_0 c) = matrixMulScalarLeft _ _
    rw [hA]; rfl
  have h10 := Fc.c0.h10_0
  have h10' := Fc.c0.h10_0
  rw [e9] at h10
  rw [e9'] at h10'
  have eD : parallel_generatedRoot_1281.w_11_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_7_0 d := by
    have h11 := Fc.c0.h11_0
    rw [gadgetDecomposeRuns_unique h10 Fd.c0.h6_0] at h11
    exact compactCenteredRebaseRuns_unique h11 Fd.c0.h7_0
  have eD' : parallel_generatedRoot_1281.w_11_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_60_0 d := by
    have h11 := Fc.c0.h11_0
    rw [gadgetDecomposeRuns_unique h10' Fd.c2.h59_0] at h11
    exact compactCenteredRebaseRuns_unique h11 Fd.c2.h60_0
  -- `Y`: the digits of `c' lift (s₁₀ s₉ (G mod q))`.
  have eY : parallel_generatedRoot_1281.w_25_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_21_0 d := by
    have eG := gadgetMatrixRuns_unique Fc.c0.h14_0 Fd.c0.h10_0
    have h15 := Fc.c0.h15_0
    rw [eG] at h15
    have e15 := centeredRebaseRuns_unique h15 Fd.c0.h11_0
    have h16 := Fc.c0.h16_0
    rw [e15] at h16
    have e16 := modulusReduceRuns_unique h16 Fd.c0.h12_0
    have e20 : parallel_generatedRoot_1281.w_20_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_16_0 d := by
      show matrixMulScalarRight (matrixMulScalarRight (parallel_generatedRoot_1281.w_16_0 c) _) _ =
        matrixMulScalarRight (matrixMulScalarRight (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_12_0 d) _) _
      rw [e16]; rfl
    have h21 := Fc.c0.h21_0
    rw [e20] at h21
    have e21 := centeredRebaseRuns_unique h21 Fd.c0.h17_0
    have e23 : parallel_generatedRoot_1281.w_23_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_19_0 d := by
      show matrixMulScalarRight (parallel_generatedRoot_1281.w_21_0 c) _ =
        matrixMulScalarRight (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_17_0 d) _
      rw [e21]; rfl
    have h24 := Fc.c0.h24_0
    rw [e23] at h24
    have h25 := Fc.c0.h25_0
    rw [gadgetDecomposeRuns_unique h24 Fd.c0.h20_0] at h25
    exact compactCenteredRebaseRuns_unique h25 Fd.c0.h21_0
  have eY' : parallel_generatedRoot_1281.w_25_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_74_0 d := by
    have eG := gadgetMatrixRuns_unique Fc.c0.h14_0 Fd.c2.h63_0
    have h15 := Fc.c0.h15_0
    rw [eG] at h15
    have e15 := centeredRebaseRuns_unique h15 Fd.c2.h64_0
    have h16 := Fc.c0.h16_0
    rw [e15] at h16
    have e16 := modulusReduceRuns_unique h16 Fd.c3.h65_0
    have e20 : parallel_generatedRoot_1281.w_20_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_69_0 d := by
      show matrixMulScalarRight (matrixMulScalarRight (parallel_generatedRoot_1281.w_16_0 c) _) _ =
        matrixMulScalarRight (matrixMulScalarRight (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_65_0 d) _) _
      rw [e16]; rfl
    have h21 := Fc.c0.h21_0
    rw [e20] at h21
    have e21 := centeredRebaseRuns_unique h21 Fd.c3.h70_0
    have e23 : parallel_generatedRoot_1281.w_23_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_72_0 d := by
      show matrixMulScalarRight (parallel_generatedRoot_1281.w_21_0 c) _ =
        matrixMulScalarRight (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_70_0 d) _
      rw [e21]; rfl
    have h24 := Fc.c0.h24_0
    rw [e23] at h24
    have h25 := Fc.c0.h25_0
    rw [gadgetDecomposeRuns_unique h24 Fd.c3.h73_0] at h25
    exact compactCenteredRebaseRuns_unique h25 Fd.c3.h74_0
  -- The fresh digits: `diag(G, …, G)` and `Q_q`.
  have dF := online1281_diagF0 Fc
  rw [hG] at dF
  have eFd : parallel_generatedRoot_1281.w_32_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_28_0 d :=
    isDiag_unique (by norm_num) (by norm_num) dF (keys1267_diagF0 Fd)
  have eFd' : parallel_generatedRoot_1281.w_32_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_80_0 d :=
    isDiag_unique (by norm_num) (by norm_num) dF (keys1267_diagF1 Fd)
  have e35 : parallel_generatedRoot_1281.w_35_0 c = parallel_generatedRoot_1281.w_32_0 c :=
    rebase_one_rows hq1 hn Fc.c1.h33_0
  have e31 : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_31_0 d = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_28_0 d := rebase_one_rows hq1 hn Fd.c1.h29_0
  have e83 : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_83_0 d = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_80_0 d := rebase_one_rows hq1 hn Fd.c4.h81_0
  have e37 : parallel_generatedRoot_1281.w_37_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_33_0 d := by
    have h36 := Fc.c1.h36_0
    rw [e35, eFd, ← e31] at h36
    have h37 := Fc.c1.h37_0
    rw [gadgetDecomposeRuns_unique h36 Fd.c1.h32_0] at h37
    exact compactCenteredRebaseRuns_unique h37 Fd.c1.h33_0
  have e37' : parallel_generatedRoot_1281.w_37_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_85_0 d := by
    have h36 := Fc.c1.h36_0
    rw [e35, eFd', ← e83] at h36
    have h37 := Fc.c1.h37_0
    rw [gadgetDecomposeRuns_unique h36 Fd.c4.h84_0] at h37
    exact compactCenteredRebaseRuns_unique h37 Fd.c4.h85_0
  have e42 : parallel_generatedRoot_1281.w_42_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_38_0 d := by
    have h41 := Fc.c1.h41_0
    have h42 := Fc.c1.h42_0
    rw [gadgetDecomposeRuns_unique h41 Fd.c1.h37_0] at h42
    exact compactCenteredRebaseRuns_unique h42 Fd.c1.h38_0
  have e42' : parallel_generatedRoot_1281.w_42_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_90_0 d := by
    have h41 := Fc.c1.h41_0
    have h42 := Fc.c1.h42_0
    rw [gadgetDecomposeRuns_unique h41 Fd.c4.h89_0] at h42
    exact compactCenteredRebaseRuns_unique h42 Fd.c4.h90_0
  -- The smudge digits: `diag(h, …, h)`.
  have dS := online1281_diagS0 Fc
  rw [hh] at dS
  have eSd : parallel_generatedRoot_1281.w_50_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_46_0 d :=
    isDiag_unique (by norm_num) (by norm_num) dS (keys1267_diagS0 Fd)
  have eSd' : parallel_generatedRoot_1281.w_50_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_97_0 d :=
    isDiag_unique (by norm_num) (by norm_num) dS (keys1267_diagS1 Fd)
  have e53 : parallel_generatedRoot_1281.w_53_0 c = parallel_generatedRoot_1281.w_50_0 c :=
    rebase_one_rows hq1 hn Fc.c2.h51_0
  have e49 : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_49_0 d = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_46_0 d := rebase_one_rows hq1 hn Fd.c2.h47_0
  have e100 : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_100_0 d = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_97_0 d := rebase_one_rows hq1 hn Fd.c5.h98_0
  have e55 : parallel_generatedRoot_1281.w_55_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_51_0 d := by
    have h54 := Fc.c2.h54_0
    rw [e53, eSd, ← e49] at h54
    have h55 := Fc.c2.h55_0
    rw [gadgetDecomposeRuns_unique h54 Fd.c2.h50_0] at h55
    exact compactCenteredRebaseRuns_unique h55 Fd.c2.h51_0
  have e55' : parallel_generatedRoot_1281.w_55_0 c = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_102_0 d := by
    have h54 := Fc.c2.h54_0
    rw [e53, eSd', ← e100] at h54
    have h55 := Fc.c2.h55_0
    rw [gadgetDecomposeRuns_unique h54 Fd.c5.h101_0] at h55
    exact compactCenteredRebaseRuns_unique h55 Fd.c5.h102_0
  -- The fresh rows.
  have eF0 : freshKey0 c hi K 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_41_0 d := by
    have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) Fd.c1.h35_0 rfl Fd.c1.h40_0
      Fd.c1.h41_0 (fun _ _ _ hb ↦ W_Stage_keys_online_0_0.sequential_parallel_generatedRoot_1267_41.sem hb)
    have ecells : cellsKey (fun cc : Fin 25 ↦ K ⟨25 * i + cc.val, by omega⟩) 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_26_0 d := by
      refine row_eq_of_blk (C := 25) (m := 25) fun cc ↦ ?_
      rw [blk_cellsKey, (hK _).1, slice_blocks (C := 900) (C' := 25) (m := 25) Fd.c0.h26_0 (25 * i)
        (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) cc (by omega)]
    funext r l
    have hr : r = 0 := Subsingleton.elim _ _
    subst hr
    rw [entry_blk (C := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_41_0 d) l, hblk]
    show (blk (C := 25) (m := 25) (cellsKey (fun cc : Fin 25 ↦ K ⟨25 * i + cc.val, by omega⟩) 0 *
      parallel_generatedRoot_1281.w_37_0 c) (RefreshTopProof.GenFinal.colOf (C := 25) l) *
        parallel_generatedRoot_1281.w_42_0 c) 0 0 = _
    rw [ecells, e37, e42]
    rfl
  have eF1 : freshKey0 c hi K 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_93_0 d := by
    have hblk := slice_column_total (C := 25) (m := 25) (by norm_num) Fd.c4.h87_0 rfl Fd.c4.h92_0
      Fd.c4.h93_0 (fun _ _ _ hb ↦ W_Stage_keys_online_0_0.sequential_parallel_generatedRoot_1267_93.sem hb)
    have ecells : cellsKey (fun cc : Fin 25 ↦ K ⟨25 * i + cc.val, by omega⟩) 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_79_0 d := by
      refine row_eq_of_blk (C := 25) (m := 25) fun cc ↦ ?_
      rw [blk_cellsKey, (hK _).2, slice_blocks (C := 900) (C' := 25) (m := 25) Fd.c3.h79_0 (25 * i)
        (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) cc (by omega)]
    funext r l
    have hr : r = 0 := Subsingleton.elim _ _
    subst hr
    rw [entry_blk (C := 25) (W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_93_0 d) l, hblk]
    show (blk (C := 25) (m := 25) (cellsKey (fun cc : Fin 25 ↦ K ⟨25 * i + cc.val, by omega⟩) 1 *
      parallel_generatedRoot_1281.w_37_0 c) (RefreshTopProof.GenFinal.colOf (C := 25) l) *
        parallel_generatedRoot_1281.w_42_0 c) 0 0 = _
    rw [ecells, e37', e42']
    rfl
  -- The smudge rows.
  have eS0 : smudgeKey0 c hi Ksm 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_52_0 d := by
    have ecells : cellsKey (fun cc : Fin 25 ↦ Ksm ⟨125 * i + cc.val, by omega⟩) 0 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_44_0 d := by
      refine row_eq_of_blk (C := 25) (m := 25) fun cc ↦ ?_
      rw [blk_cellsKey, (hKs _).1, slice_blocks (C := 4500) (C' := 25) (m := 25) Fd.c1.h44_0 (125 * i)
        (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) cc (by omega)]
    show cellsKey (fun cc : Fin 25 ↦ Ksm ⟨125 * i + cc.val, by omega⟩) 0 *
      parallel_generatedRoot_1281.w_55_0 c = _
    rw [ecells, e55]
    rfl
  have eS1 : smudgeKey0 c hi Ksm 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_103_0 d := by
    have ecells : cellsKey (fun cc : Fin 25 ↦ Ksm ⟨125 * i + cc.val, by omega⟩) 1 = W_Stage_keys_online_0_0.parallel_generatedRoot_1267.w_96_0 d := by
      refine row_eq_of_blk (C := 25) (m := 25) fun cc ↦ ?_
      rw [blk_cellsKey, (hKs _).2, slice_blocks (C := 4500) (C' := 25) (m := 25) Fd.c4.h96_0 (125 * i)
        (by simp only [Int.ofNat_eq_natCast]; push_cast; ring) cc (by omega)]
    show cellsKey (fun cc : Fin 25 ↦ Ksm ⟨125 * i + cc.val, by omega⟩) 1 *
      parallel_generatedRoot_1281.w_55_0 c = _
    rw [ecells, e55']
    rfl
  have k0 : MxxExponentLut.rowOf A 0 * parallel_generatedRoot_1281.w_1_0 c =
      (Kg 0 * topRows (parallel_generatedRoot_1281.w_11_0 c) + K₁ 0 * bottomRows (parallel_generatedRoot_1281.w_11_0 c) -
        Kx 0 * parallel_generatedRoot_1281.w_25_0 c) + freshKey0 c hi K 0 + smudgeKey0 c hi Ksm 0 := by
    rw [rowOf_mul_of_solved hP 0, r0, hKg.1, hK₁.1, hKx.1, eD, eY, eF0, eS0]
  have k1 : MxxExponentLut.rowOf A 1 * parallel_generatedRoot_1281.w_1_0 c =
      (Kg 1 * topRows (parallel_generatedRoot_1281.w_11_0 c) + K₁ 1 * bottomRows (parallel_generatedRoot_1281.w_11_0 c) -
        Kx 1 * parallel_generatedRoot_1281.w_25_0 c) + freshKey0 c hi K 1 + smudgeKey0 c hi Ksm 1 := by
    rw [rowOf_mul_of_solved hP 1, r1, hKg.2, hK₁.2, hKx.2, eD', eY', eF1, eS1]
  intro k'
  fin_cases k'
  exacts [k0, k1]

end PartAgree

/-! ## Stage links -/

/-- A packed family is determined by its list. -/
theorem familyPack_eq {α : Type} {size : Nat} {c c' : Int} {l l' : List α} {f f' : Fin size → α}
    (h : familyPack c l f) (h' : familyPack c' l' f') (hl : l = l') : f = f' := by
  subst hl
  funext i
  obtain ⟨-, -, h⟩ := h
  obtain ⟨-, -, h'⟩ := h'
  obtain ⟨p, hp, e⟩ := h i
  obtain ⟨p', hp', e'⟩ := h' i
  rw [e, e']
  congr 1
  exact Fin.ext (by omega)

@@LINK@@
@@PUBFAM@@
@@GADGET@@

/-- The online keys of the σ carrier `g` of step 0 (`online0_g`): component `1` of the σ sampler
total's keys times the digits of the gadget. -/
noncomputable def sigmaKey0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} (Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) (k : Fin 2) :
    ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  ![![W_Stage_online_0.generatedRoot.w_56_0 Con, W_Stage_online_0.generatedRoot.w_56_1 Con], ![W_Stage_online_0.generatedRoot.w_56_2 Con, W_Stage_online_0.generatedRoot.w_56_3 Con]] k 1 *
    W_Stage_online_0.generatedRoot.w_62_0 Con

/-- The online keys of the one-public row `c₁` of step 0 (`online0_c1` with the root keys of label
member `1`). -/
noncomputable def onePubKey0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} (Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) (k : Fin 2) :
    ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  labelKeysG hashModel (W_Stage_online_0.generatedRoot.w_7_0 Con) W_Stage_online_0.modulus_0 25 labelPre 1 k 1 *
    W_Stage_online_0.generatedRoot.w_69_0 Con

/-- The fresh cell keys of step 0: block `cc` of the keys `A_{k,1}` of the fresh column total
(`online0_fresh_row`). -/
noncomputable def freshCells0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} (Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) (cc : Fin 900)
    (k : Fin 2) : ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  blk (C := 900) (m := 25) (![W_Stage_online_0.generatedRoot.w_1219_1 Con, W_Stage_online_0.generatedRoot.w_1219_3 Con] k) cc

/-- The smudge cell keys of step 0: block `cc` of the keys `A_{k,1}` of the smudge column total. -/
noncomputable def smudgeCells0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} (Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) (cc : Fin 4500)
    (k : Fin 2) : ExactMatrix W_Stage_online_0.modulus_0 8 1 25 :=
  blk (C := 4500) (m := 25) (![W_Stage_online_0.generatedRoot.w_1273_1 Con, W_Stage_online_0.generatedRoot.w_1273_3 Con] k) cc

set_option maxHeartbeats 64000000 in
/-- **`hpre` of `online0_part_prime0` at the stage level, from the key agreements.** Part `i` of
the online stage of step 0 (`c`, with the inputs the stage passes it) and the keys stage (`Ck`,
linked by `L`): if the preimages solve against the keys stage's prime-0 outputs (`hsolve`), and
the σ sampler total, the fresh and smudge column totals have the same keys `A_{k,1}` in both
stages (`hSig`, `hF`, `hS`) and the online `cₓ` keys `Kx` are the keys stage's two `cₓ` rows (`hKx`),
then `hpre` holds with the online keys `sigmaKey0`, `onePubKey0`, `Kx`, `freshKey0`,
`smudgeKey0`. -/
theorem online0_hpre0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} {paramsK : W_Stage_keys_online_0_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params} {oon : _} (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _} (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok)
    (L : Link0 Con Ck) (hq1 : 1 < W_Stage_online_0.modulus_0)
    {A : ExactMatrix W_Stage_online_0.modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * W_Stage_online_0.generatedRoot.w_1_0 Con j = W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck j)
    (hSig : W_Stage_online_0.generatedRoot.w_56_1 Con = W_Stage_keys_online_0_0.generatedRoot.w_54_1 Ck ∧ W_Stage_online_0.generatedRoot.w_56_3 Con = W_Stage_keys_online_0_0.generatedRoot.w_54_3 Ck)
    (hF : W_Stage_online_0.generatedRoot.w_1219_1 Con = W_Stage_keys_online_0_0.generatedRoot.w_1114_1 Ck ∧ W_Stage_online_0.generatedRoot.w_1219_3 Con = W_Stage_keys_online_0_0.generatedRoot.w_1114_3 Ck)
    (hS : W_Stage_online_0.generatedRoot.w_1273_1 Con = W_Stage_keys_online_0_0.generatedRoot.w_1168_1 Ck ∧ W_Stage_online_0.generatedRoot.w_1273_3 Con = W_Stage_keys_online_0_0.generatedRoot.w_1168_3 Ck)
    {Kx : Fin 36 → Fin 2 → ExactMatrix W_Stage_online_0.modulus_0 8 1 25}
    (hKx : ∀ j, Kx j 0 = W_Stage_keys_online_0_0.generatedRoot.w_1057_0 Ck j ∧ Kx j 1 = W_Stage_keys_online_0_0.generatedRoot.w_1266_0 Ck j)
    (i : Fin 36) (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = (W_Stage_online_0.generatedRoot.w_0_0 Con, ((W_Stage_online_0.generatedRoot.w_1_0 Con i), (W_Stage_online_0.generatedRoot.w_63_0 Con, (W_Stage_online_0.generatedRoot.w_70_0 Con,
      ((W_Stage_online_0.generatedRoot.w_227_0 Con i), ((W_Stage_online_0.generatedRoot.w_1162_0 Con i), (W_Stage_online_0.generatedRoot.w_1219_5 Con, (W_Stage_online_0.generatedRoot.w_57_0 Con,
      (W_Stage_online_0.generatedRoot.w_1273_5 Con, (W_Stage_online_0.generatedRoot.w_1274_0 Con, (W_Stage_online_0.generatedRoot.w_1276_0 Con, ((W_Stage_online_0.generatedRoot.w_1277_0 Con i),
      ((W_Stage_online_0.generatedRoot.w_1278_0 Con i), ((W_Stage_online_0.generatedRoot.w_1279_0 Con i), ((W_Stage_online_0.generatedRoot.w_1280_0 Con i), ()))))))))))))))))
    {oc : _} (Fc : W_Stage_online_0.parallel_generatedRoot_1281.Facts c oc) :
    ∀ k', MxxExponentLut.rowOf A k' * W_Stage_online_0.parallel_generatedRoot_1281.w_1_0 c =
      (sigmaKey0 Con k' * topRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) +
        onePubKey0 Con k' * bottomRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) -
        Kx i k' * W_Stage_online_0.parallel_generatedRoot_1281.w_25_0 c) +
        freshKey0 c i.isLt (freshCells0 Con) k' + smudgeKey0 c i.isLt (smudgeCells0 Con) k' := by
  obtain ⟨eG, eD, eD1⟩ := gadget_agree0 Fon Fk
  have eA := pub_family_agree0 Fon Fk L
  -- The keys stage's part `i`.
  obtain ⟨wd, hd⟩ := Fk.c49.h1267_0 i
  let d : W_Stage_keys_online_0_0.parallel_generatedRoot_1267.Ctx backend hashModel paramsK i :=
    ⟨(W_Stage_keys_online_0_0.generatedRoot.w_61_0 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_68_0 Ck, ((W_Stage_keys_online_0_0.generatedRoot.w_225_0 Ck i), ((W_Stage_keys_online_0_0.generatedRoot.w_1057_0 Ck i), (W_Stage_keys_online_0_0.generatedRoot.w_1114_1 Ck,
      (W_Stage_keys_online_0_0.generatedRoot.w_55_0 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_1168_1 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_1169_0 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_400_0 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_408_0 Ck,
      ((W_Stage_keys_online_0_0.generatedRoot.w_1266_0 Ck i), (W_Stage_keys_online_0_0.generatedRoot.w_1114_3 Ck, (W_Stage_keys_online_0_0.generatedRoot.w_1168_3 Ck, ()))))))))))))), wd⟩
  have Fd := W_Stage_keys_online_0_0.parallel_generatedRoot_1267.facts d hd
  have hP : A * W_Stage_online_0.parallel_generatedRoot_1281.w_1_0 c = W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck i := by
    show A * c.inputs.2.1 = _
    rw [hc]
    exact hsolve i
  -- The label keys of the one-public row.
  have hl0 : W_Stage_keys_online_0_0.generatedRoot.w_62_0 Ck = labelKeysG hashModel (W_Stage_keys_online_0_0.generatedRoot.w_5_0 Ck) W_Stage_keys_online_0_0.modulus_0 25
      labelPre 1 0 1 :=
    keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 0 1 (by norm_num) Fk.c1.h62_0
  have hl1 : W_Stage_keys_online_0_0.generatedRoot.w_407_0 Ck = labelKeysG hashModel (W_Stage_keys_online_0_0.generatedRoot.w_5_0 Ck) W_Stage_keys_online_0_0.modulus_0 25
      labelPre 1 1 1 :=
    keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 1 1 (by norm_num) Fk.c14.h407_0
  refine part_agree0 i.isLt c Fc d Fd hq1 ?_ ?_ ?_ hP ?_ ?_ ?_ ?_ ?_
  · show c.inputs.2.2.2.2.1 = _
    rw [hc]
    exact congrFun eA i
  · show c.inputs.2.2.2.2.2.2.2.1 = _
    rw [hc]
    exact eG
  · show c.inputs.2.2.2.2.2.2.2.2.2.1 = _
    rw [hc]
    rfl
  · refine ⟨?_, ?_⟩
    · show W_Stage_online_0.generatedRoot.w_56_1 Con * W_Stage_online_0.generatedRoot.w_62_0 Con = W_Stage_keys_online_0_0.generatedRoot.w_54_1 Ck * W_Stage_keys_online_0_0.generatedRoot.w_60_0 Ck
      rw [hSig.1, eD]
    · show W_Stage_online_0.generatedRoot.w_56_3 Con * W_Stage_online_0.generatedRoot.w_62_0 Con = W_Stage_keys_online_0_0.generatedRoot.w_54_3 Ck * W_Stage_keys_online_0_0.generatedRoot.w_60_0 Ck
      rw [hSig.2, eD]
  · refine ⟨?_, ?_⟩
    · show _ * W_Stage_online_0.generatedRoot.w_69_0 Con = W_Stage_keys_online_0_0.generatedRoot.w_62_0 Ck * W_Stage_keys_online_0_0.generatedRoot.w_67_0 Ck
      rw [hl0, eD1, L.e7]
    · show _ * W_Stage_online_0.generatedRoot.w_69_0 Con = W_Stage_keys_online_0_0.generatedRoot.w_407_0 Ck * W_Stage_keys_online_0_0.generatedRoot.w_67_0 Ck
      rw [hl1, eD1, L.e7]
  · exact hKx i
  · intro cc
    refine ⟨?_, ?_⟩
    · show blk (C := 900) (m := 25) (W_Stage_online_0.generatedRoot.w_1219_1 Con) cc = _
      rw [hF.1]
      rfl
    · show blk (C := 900) (m := 25) (W_Stage_online_0.generatedRoot.w_1219_3 Con) cc = _
      rw [hF.2]
      rfl
  · intro cc
    refine ⟨?_, ?_⟩
    · show blk (C := 4500) (m := 25) (W_Stage_online_0.generatedRoot.w_1273_1 Con) cc = _
      rw [hS.1]
      rfl
    · show blk (C := 4500) (m := 25) (W_Stage_online_0.generatedRoot.w_1273_3 Con) cc = _
      rw [hS.2]
      rfl

/-! ## The key agreement of the PRF chains -/

@@CHAINS@@

set_option maxHeartbeats 64000000 in
/-- **`hpre` of `online0_part_prime0` at the stage level.** `online0_hpre0` with the σ sampler,
fresh and smudge agreements discharged (`sigma_total_agree0`, `fresh_total_agree0`,
`smudge_total_agree0`): it remains that the preimages solve against the keys stage's prime-0
outputs (`hsolve`) and that the online `cₓ` keys are the keys stage's two `cₓ` rows (`hKx`). -/
theorem online0_hpre0' {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} {paramsK : W_Stage_keys_online_0_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params} {oon : _} (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _} (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok)
    (L : Link0 Con Ck) (hq1 : 1 < W_Stage_online_0.modulus_0)
    (layout : RegularLayout W_Stage_online_0.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_0.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix W_Stage_online_0.modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * W_Stage_online_0.generatedRoot.w_1_0 Con j = W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck j)
    {Kx : Fin 36 → Fin 2 → ExactMatrix W_Stage_online_0.modulus_0 8 1 25}
    (hKx : ∀ j, Kx j 0 = W_Stage_keys_online_0_0.generatedRoot.w_1057_0 Ck j ∧ Kx j 1 = W_Stage_keys_online_0_0.generatedRoot.w_1266_0 Ck j)
    (i : Fin 36) (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = (W_Stage_online_0.generatedRoot.w_0_0 Con, ((W_Stage_online_0.generatedRoot.w_1_0 Con i), (W_Stage_online_0.generatedRoot.w_63_0 Con, (W_Stage_online_0.generatedRoot.w_70_0 Con,
      ((W_Stage_online_0.generatedRoot.w_227_0 Con i), ((W_Stage_online_0.generatedRoot.w_1162_0 Con i), (W_Stage_online_0.generatedRoot.w_1219_5 Con, (W_Stage_online_0.generatedRoot.w_57_0 Con,
      (W_Stage_online_0.generatedRoot.w_1273_5 Con, (W_Stage_online_0.generatedRoot.w_1274_0 Con, (W_Stage_online_0.generatedRoot.w_1276_0 Con, ((W_Stage_online_0.generatedRoot.w_1277_0 Con i),
      ((W_Stage_online_0.generatedRoot.w_1278_0 Con i), ((W_Stage_online_0.generatedRoot.w_1279_0 Con i), ((W_Stage_online_0.generatedRoot.w_1280_0 Con i), ()))))))))))))))))
    {oc : _} (Fc : W_Stage_online_0.parallel_generatedRoot_1281.Facts c oc) :
    ∀ k', MxxExponentLut.rowOf A k' * W_Stage_online_0.parallel_generatedRoot_1281.w_1_0 c =
      (sigmaKey0 Con k' * topRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) +
        onePubKey0 Con k' * bottomRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) -
        Kx i k' * W_Stage_online_0.parallel_generatedRoot_1281.w_25_0 c) +
        freshKey0 c i.isLt (freshCells0 Con) k' + smudgeKey0 c i.isLt (smudgeCells0 Con) k' := by
  obtain ⟨-, s1, -, s3⟩ := sigma_total_agree0 Fon Fk L hq1 layout hlayout hexact
  obtain ⟨-, f1, -, f3⟩ := fresh_total_agree0 Fon Fk L hq1 layout hlayout hexact
  obtain ⟨-, m1, -, m3⟩ := smudge_total_agree0 Fon Fk L hq1 layout hlayout hexact
  exact online0_hpre0 Fon Fk L hq1 hsolve ⟨s1, s3⟩ ⟨f1, f3⟩ ⟨m1, m3⟩ hKx i c hc Fc

/-! ## The `cₓ` keys -/

/-- A member of a packed family is the list's member. -/
theorem familyPack_get {α : Type} {size : Nat} {c : Int} {l : List α} {f : Fin size → α}
    (h : familyPack c l f) (j : Fin size) : ∃ hj : j.val < l.length, f j = l.get ⟨j.val, hj⟩ := by
  obtain ⟨-, -, h⟩ := h
  obtain ⟨p, hp, e⟩ := h j
  refine ⟨hp ▸ p.isLt, ?_⟩
  rw [e]
  congr 1
  exact Fin.ext hp


@@STATE@@
@@CX@@

set_option maxHeartbeats 64000000 in
/-- **`hpre` of `online0_part_prime0` at the stage level, all key agreements discharged.** With
the online `cₓ` keys `cxKeys0` (`cx_keys_agree0`), it remains only that the preimages solve
against the keys stage's prime-0 outputs (`hsolve`, from `claim_link0`). -/
theorem online0_hpre0_full {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params} {paramsK : W_Stage_keys_online_0_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend hashModel params} {oon : _} (Fon : W_Stage_online_0.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx backend hashModel paramsK} {ok : _} (Fk : W_Stage_keys_online_0_0.generatedRoot.Facts Ck ok)
    (L : Link0 Con Ck) (hq1 : 1 < W_Stage_online_0.modulus_0)
    (layout : RegularLayout W_Stage_online_0.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_0.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    {A : ExactMatrix W_Stage_online_0.modulus_0 8 2 54}
    (hsolve : ∀ j : Fin 36, A * W_Stage_online_0.generatedRoot.w_1_0 Con j = W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck j)
    (i : Fin 36) (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params i)
    (hc : c.inputs = (W_Stage_online_0.generatedRoot.w_0_0 Con, ((W_Stage_online_0.generatedRoot.w_1_0 Con i), (W_Stage_online_0.generatedRoot.w_63_0 Con, (W_Stage_online_0.generatedRoot.w_70_0 Con,
      ((W_Stage_online_0.generatedRoot.w_227_0 Con i), ((W_Stage_online_0.generatedRoot.w_1162_0 Con i), (W_Stage_online_0.generatedRoot.w_1219_5 Con, (W_Stage_online_0.generatedRoot.w_57_0 Con,
      (W_Stage_online_0.generatedRoot.w_1273_5 Con, (W_Stage_online_0.generatedRoot.w_1274_0 Con, (W_Stage_online_0.generatedRoot.w_1276_0 Con, ((W_Stage_online_0.generatedRoot.w_1277_0 Con i),
      ((W_Stage_online_0.generatedRoot.w_1278_0 Con i), ((W_Stage_online_0.generatedRoot.w_1279_0 Con i), ((W_Stage_online_0.generatedRoot.w_1280_0 Con i), ()))))))))))))))))
    {oc : _} (Fc : W_Stage_online_0.parallel_generatedRoot_1281.Facts c oc) :
    ∀ k', MxxExponentLut.rowOf A k' * W_Stage_online_0.parallel_generatedRoot_1281.w_1_0 c =
      (sigmaKey0 Con k' * topRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) +
        onePubKey0 Con k' * bottomRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) -
        cxKeys0 Con i k' * W_Stage_online_0.parallel_generatedRoot_1281.w_25_0 c) +
        freshKey0 c i.isLt (freshCells0 Con) k' + smudgeKey0 c i.isLt (smudgeCells0 Con) k' :=
  online0_hpre0' Fon Fk L hq1 layout hlayout hexact hsolve
    (cx_keys_agree0 Fon Fk L hq1 layout hlayout hexact) i c hc Fc

/-! ## The claim-level wiring -/

set_option maxHeartbeats 64000000 in
/-- **The claim-level wiring of step 0's key agreement.** Under `W_GeneratedClaim.Runs`, the online
stage of step 0 (stage `11`) and its keys stage (stage `8`) run on linked inputs (`Link0`: every
input both read from the same source), and the step-0 preimages (stage `9`, the online stage's
input `1`) solve against the anchor `B_0` and the keys stage's prime-0 output family. -/
theorem claim_link0 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) :
    ∃ (Con : W_Stage_online_0.generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params)
      (Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_8_params),
      W_Stage_online_0.generatedRoot.Facts Con execution.«stage_11» ∧ W_Stage_keys_online_0_0.generatedRoot.Facts Ck execution.«stage_8» ∧ Link0 Con Ck ∧
      (∀ j : Fin 36, execution.«stage_1».1 * W_Stage_online_0.generatedRoot.w_1_0 Con j = W_Stage_keys_online_0_0.generatedRoot.w_1267_0 Ck j) := by
  have hp := (claim_preimages h).1.1
  obtain ⟨-, -, -, -, -, -, -, -, -, h8, -, -, h11, -⟩ := h
  obtain ⟨wk, hk⟩ := h8
  obtain ⟨wo, ho⟩ := h11
  have Fk := W_Stage_keys_online_0_0.generatedRoot.facts ⟨_, wk⟩ hk
  have Fo := W_Stage_online_0.generatedRoot.facts ⟨_, wo⟩ ho
  refine ⟨⟨_, wo⟩, ⟨_, wk⟩, Fo, Fk, ⟨@@RFL@@⟩, fun j ↦ ?_⟩
  have e := (hp j).1
  rw [Fk.c67.hout] at e
  exact e

/-- **The anchor link.** The anchor the obfuscator publishes and the initial stage encrypts
(output `0` of `anchor_publics`) is the sampled anchor `B_0` (output `0` of `anchors`). -/
theorem claim_anchor0 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) :
    execution.«stage_2».1 = execution.«stage_1».1 := by
  obtain ⟨-, -, h1, h2, -⟩ := h
  have hA := (anchors_trapdoor h1).1
  obtain ⟨_, hb⟩ := h2
  rw [show execution.«stage_2» = _ from hb]
  exact trapdoorSample_public hA

/-- **The hint of step 0 under the anchor's rows.** The online stage's input `0` (the hint
`b_0`, output `0` of the initial stage) is `s̄_0 B_0[0] + B_0[1] + e` for the sampled anchor
`B_0`: the keys `Bk` of `online0_part_prime0`'s `hb` are the rows of `B_0`, the matrix the
preimages solve against (`claim_link0`). -/
theorem claim_hint0 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) {sm : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm) :
    ∃ e : ErrorMatrix 8 1 54, CoeffBound e 1 ∧
      execution.«stage_4».1 = maskPart (reducePoly W_Stage_initial.modulus_0 8 sm)
        (MxxExponentLut.rowOf execution.«stage_1».1) + reduceMatrix W_Stage_initial.modulus_0 8 1 54 e := by
  have hA := claim_anchor0 h
  obtain ⟨-, -, -, -, -, h4, -⟩ := h
  obtain ⟨e, he, hh⟩ := initial_hint h4 hmask
  refine ⟨e, he, ?_⟩
  rw [hh, ← hA]
  rfl

set_option maxHeartbeats 64000000 in
/-- **`hpre` of `online0_part_prime0` at the claim level.** Under `W_GeneratedClaim.Runs`, the
online stage of step 0 (`Con`, the execution's stage `11`) and its keys stage (`Ck`, stage `8`)
satisfy: for every part `i` (its context `c` reading the inputs the stage passes it) and both
mask components `k`, row `k` of the anchor `B_0` times the part's preimage is the online
combination of keys `(K_g,k D_top + K₁,k D_bot - K_x,k Y) + K_f,k + K_s,k` with the online keys
`sigmaKey0`, `onePubKey0`, `cxKeys0 Con i`, `freshKey0 … freshCells0`, `smudgeKey0 … smudgeCells0`
(the forms of `online0_g`, `online0_c1`, the target theorems, `online0_part_fresh0` and
`online0_part_smudge0`). With `claim_hint0`, `Bk = rowOf B_0`. -/
theorem claim_hpre0 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution)
    (hq1 : 1 < W_Stage_online_0.modulus_0) (layout : RegularLayout W_Stage_online_0.modulus_0)
    (hlayout : Backend.backend.regularLayout W_Stage_online_0.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ (Con : W_Stage_online_0.generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_11_params)
      (Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx Backend.backend hashModel W_GeneratedClaim.stage_8_params),
      W_Stage_online_0.generatedRoot.Facts Con execution.«stage_11» ∧ W_Stage_keys_online_0_0.generatedRoot.Facts Ck execution.«stage_8» ∧ Link0 Con Ck ∧
      ∀ (i : Fin 36)
        (c : W_Stage_online_0.parallel_generatedRoot_1281.Ctx Backend.backend hashModel
          W_GeneratedClaim.stage_11_params i),
        c.inputs = (W_Stage_online_0.generatedRoot.w_0_0 Con, ((W_Stage_online_0.generatedRoot.w_1_0 Con i), (W_Stage_online_0.generatedRoot.w_63_0 Con, (W_Stage_online_0.generatedRoot.w_70_0 Con,
          ((W_Stage_online_0.generatedRoot.w_227_0 Con i), ((W_Stage_online_0.generatedRoot.w_1162_0 Con i), (W_Stage_online_0.generatedRoot.w_1219_5 Con, (W_Stage_online_0.generatedRoot.w_57_0 Con,
          (W_Stage_online_0.generatedRoot.w_1273_5 Con, (W_Stage_online_0.generatedRoot.w_1274_0 Con, (W_Stage_online_0.generatedRoot.w_1276_0 Con, ((W_Stage_online_0.generatedRoot.w_1277_0 Con i),
          ((W_Stage_online_0.generatedRoot.w_1278_0 Con i), ((W_Stage_online_0.generatedRoot.w_1279_0 Con i), ((W_Stage_online_0.generatedRoot.w_1280_0 Con i), ()))))))))))))))) →
        ∀ {oc : _}, W_Stage_online_0.parallel_generatedRoot_1281.Facts c oc →
        ∀ k', MxxExponentLut.rowOf execution.«stage_1».1 k' *
            W_Stage_online_0.parallel_generatedRoot_1281.w_1_0 c =
          (sigmaKey0 Con k' * topRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) +
            onePubKey0 Con k' * bottomRows (W_Stage_online_0.parallel_generatedRoot_1281.w_11_0 c) -
            cxKeys0 Con i k' * W_Stage_online_0.parallel_generatedRoot_1281.w_25_0 c) +
            freshKey0 c i.isLt (freshCells0 Con) k' + smudgeKey0 c i.isLt (smudgeCells0 Con) k' := by
  obtain ⟨Con, Ck, Fo, Fk, L, hs⟩ := claim_link0 h
  exact ⟨Con, Ck, Fo, Fk, L, fun i c hc _ Fc ↦
    online0_hpre0_full Fo Fk L hq1 layout hlayout hexact hs i c hc Fc⟩

end Dio2Proof
"""


def main():
    pairs = link_pairs()
    links.update({(w, kw): f'L.e{w.split("_")[1]}' for w, kw in pairs})
    chains = ('open RefreshTopProof.GenIface in\n' + fresh_lemma() + '\nopen RefreshTopProof.GenIface in\n' +
              sigma() + '\n\nopen RefreshTopProof.GenIface in\n' + smudge() + '\n')
    given = {('w_56_0', 'w_54_0'): 's0', ('w_56_1', 'w_54_1'): 's1', ('w_56_2', 'w_54_2'): 's2',
             ('w_56_3', 'w_54_3'): 's3'}
    state = lut_loop_lemma(317, 315, 'h317_9', 'h315_9', 'W_Stage_online_0.sequential_generatedRoot_317',
                           'W_Stage_keys_online_0_0.sequential_generatedRoot_315',
                           ['w_305_0', 'w_306_0', 'w_315_0', 'w_316_0'], ['w_303_0', 'w_304_0', 'w_313_0', 'w_314_0'],
                           ['w_247_0', 'w_249_0', 'w_251_0', 'w_253_0'], ['w_245_0', 'w_247_0', 'w_249_0', 'w_251_0'],
                           'w_7_0', 'w_5_0', '[(1), (3), (5), (15)]',
                           '[114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]', given)
    sub = {
        'DIAG': diag_lemmas() + '\n',
        'LINK': link_structure(pairs),
        'PUBFAM': fam_lemma('pub_family_agree0', "**The parts' public matrices agree**: the family `A'` (`w_227`, the root keys of the 36 state rows, hashed under the root-input key) is the keys stage's `w_225`.", 'w_227_0', 'w_225_0', 'h227_0', 'h225_0') + '\n',
        'GADGET': gadget_lemma(),
        'CHAINS': chains,
        'STATE': 'open RefreshTopProof.GenIface in\n' + state + '\n\n',
        'CX': 'open RefreshTopProof.GenIface in\n' + cx_lemma() + '\n',
        'RFL': ', '.join(['rfl'] * len(pairs)),
    }
    out = TEMPLATE
    for k, v in sub.items():
        out = out.replace('@@' + k + '@@\n' if k != 'RFL' else '@@RFL@@', v)
    assert '@@' not in out
    out = re.sub(r'\n{3,}', '\n\n', out)
    path = os.path.join(PKG, 'proof', 'Dio2KeyAgree0.lean')
    open(path, 'w').write(out)
    print(path, out.count('\n'), 'lines')


if __name__ == '__main__':
    main()
