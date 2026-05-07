---
name: aii-lean
description: Compiles and verifies Lean 4 code using lean-interact. Use for checking proof validity, theorem proving, and formal verification. Supports search across Mathlib, tactic suggestions (exact?, apply?, simp?), and sorry-driven proof development.
---

**IMPORTANT - Path resolution:** Always use an absolute SKILL_DIR. The CWD may not be the project root (e.g. on worker pods).
```
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```
GNU `parallel` subshells do NOT inherit `source activate`. Use `export` and **single-quoted** command templates.

## Workflow: Sorry-Driven Proof Development

The standard mathematician workflow for formalizing proofs in Lean 4:

### Step 1: Formalize the Statement
Write the theorem signature — what you want to prove:
```lean
import Mathlib.Tactic

theorem my_theorem (x y : Int) (h : x < y) : x + 1 ≤ y := by
  sorry
```

### Step 2: Search Mathlib for Relevant Lemmas
Find existing theorems by type pattern via Loogle:
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_mathlib_pattern_search.py "Int.lt_iff_add_one_le"
```

### Step 3: Try Tactics at Sorry Positions
Submit code with sorry and let the suggest tool try tactics:
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_lean_suggest.py \
  --code "import Mathlib.Tactic
theorem ex : 1 + 1 = 2 := by sorry" \
  --tactics "exact?,simp?,omega,ring"
```

Returns goals at each sorry and which tactics close them.

### Step 4: Fill Sorrys Iteratively
Replace each sorry with the tactic that worked. Sorrys can be filled in any order — each is independent. For complex proofs, break into sub-lemmas with their own sorrys.

### Step 5: Verify Complete Proof
Compile the full proof — a clean compile with no sorrys means done:
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
echo 'import Mathlib.Tactic
theorem ex (x y : Int) (h : x < y) : x + 1 ≤ y := by linarith' | $SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_run_lean.py -
```

`verified: true` = proof is complete and correct.

---

## Scripts

### Run / Verify (aii_run_lean.py)

Compile and verify Lean 4 code. Mathlib always enabled. Returns JSON with goal states at sorry positions.

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
echo 'theorem test : 1 + 1 = 2 := rfl' | $SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_run_lean.py -
```

**Parallel execution:**
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_run_lean.py" && \
parallel -j 30 -k --group --will-cite '$PY $S {}' ::: proof1.lean proof2.lean
```

**Output (verified):**
```json
{
  "success": true,
  "verified": true,
  "has_sorries": false,
  "sorry_goals": [],
  "errors": [],
  "warnings": [],
  "infos": []
}
```

**Output (sorry — shows goals):**
```json
{
  "success": true,
  "verified": false,
  "has_sorries": true,
  "sorry_goals": [
    {"sorry_index": 0, "goal": "⊢ 1 + 1 = 2", "proof_state": 0}
  ],
  "errors": [],
  "warnings": ["declaration uses 'sorry'"],
  "infos": []
}
```

**Parameters:**
- `file` (required) — Lean file to verify, or `-` for stdin
- Exit code 0 = verified, 1 = failed

---

### Suggest Tactics (aii_lean_suggest.py)

Try tactics at sorry positions. Extracts goals, runs each tactic, reports what works.

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_lean_suggest.py \
  --code "import Mathlib.Tactic
theorem ex : 1 + 1 = 2 := by sorry" \
  --tactics "exact?,simp?,omega,ring"
```

**Output:**
```json
{
  "success": true,
  "goals": [
    {"sorry_index": 0, "goal": "⊢ 1 + 1 = 2", "proof_state": 0}
  ],
  "suggestions": [
    {"sorry_index": 0, "tactic": "exact?", "success": true, "result": "Try this: exact rfl", "closes_goal": true},
    {"sorry_index": 0, "tactic": "simp?", "success": true, "result": "Try this: simp", "closes_goal": true},
    {"sorry_index": 0, "tactic": "omega", "success": true, "result": "", "closes_goal": true}
  ],
  "errors": []
}
```

**Parameters:**
- `--code, -c` (required) — Lean 4 code with sorry placeholders
- `--tactics, -t` (optional) — Comma-separated tactics (default: exact?,apply?,simp?,rw?,simp,aesop,omega,decide,ring,linarith,nlinarith,norm_num,field_simp,positivity)

**Useful tactics to try:**
- Discovery: `exact?`, `apply?`, `rw?`, `simp?`
- Automation: `simp`, `aesop`, `omega`, `decide`, `ring`, `linarith`, `nlinarith`, `norm_num`
- Field: `field_simp`, `polyrith`

---

### Pattern Search (aii_mathlib_pattern_search.py)

Search by type signature and patterns via Loogle API.

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_mathlib_pattern_search.py "List.map"
```

**Parallel execution:**
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-lean" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_mathlib_pattern_search.py" && \
parallel -j 50 -k --group --will-cite '$PY $S {} --limit 10' ::: 'List.map' 'Nat.Prime'
```

**Query patterns:**
- By constant: `Real.sin`
- By name substring: `"differ"`
- By subexpression: `_ * (_ ^ _)`
- Non-linear: `Real.sqrt ?a * Real.sqrt ?a`
- By conclusion: `|- tsum _ = _ * tsum _`
- Multiple filters: `Real.sin, "two", _ * _`

**Parameters:**
- `query` (required) — Type pattern query
- `--limit, -n` — Number of results (default: 10)
- `--timeout, -t` — Timeout in seconds (default: 30)

**Tip:** Pure type queries like `Nat → Nat` timeout — add a constant: `Nat.succ, Nat → Nat`

---

## Mathlib Tactics Reference

Mathlib is always enabled (Lean v4.14.0). Common tactics:

**Automation (close goals directly):**
- `ring` — Polynomial ring equations
- `linarith` — Linear arithmetic over ordered fields
- `nlinarith` — Nonlinear arithmetic
- `omega` — Integer/natural linear arithmetic (decision procedure)
- `decide` — Decidable propositions
- `norm_num` — Numeric normalization
- `simp` — Simplifier with extensible lemma set
- `aesop` — General proof search (best-first)

**Discovery (find what lemma/tactic to use):**
- `exact?` — Find a single lemma that closes the goal
- `apply?` — Find a lemma that applies (may leave subgoals)
- `rw?` — Find rewrite lemmas for subterms
- `simp?` — Run simplifier and report which lemmas fired

**Examples:**
```lean
import Mathlib.Tactic

-- ring closes polynomial identities
example (x y : Int) : (x + y)^2 = x^2 + 2*x*y + y^2 := by ring

-- linarith closes linear inequalities
example (x y : Int) (h1 : x < y) (h2 : y < x + 3) : y - x < 3 := by linarith

-- omega handles Nat/Int linear arithmetic
example (n : Nat) (h : n ≥ 2) : n * n ≥ 4 := by omega
```

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
