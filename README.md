# JunctionRelay_LLMSuiteKit

What every LLM*Test suite shares. Two files:

| File | Travels? | What |
|---|---|---|
| `jrsuite.py` | **yes** - copy it into any suite, stdlib only | posts a suite's results to JunctionRelay: the suite's self-description, one model's rows all-or-nothing, each confirmed by the server |
| `labtargets.py` | with your `targets.json` | your boxes (`targets.json`, gitignored; copy `targets.example.json`) plus the hosted models (`TARGETS`), the one `ask_target()` door to them, and the setup the running server reports (`runtime_facts`, `container_facts`) |

Clone the suites side by side; each loads this kit from the sibling `JunctionRelay_LLMSuiteKit`
checkout (`JR_SUITEKIT` names another path): JunctionRelay_LLMAgentTest, _LLMStandardTest,
_LLMDelegationTest, _LLMResearchTest. JunctionRelay_LLMSpeedTest keeps its own posting;
`labtargets` imports it from its sibling checkout (`JR_SPEEDTEST`) for the container setup capture.

## Bring your own suite

JR keeps no list of suites. Everything its Benchmarks page shows about yours comes from what you
post (JR `docs/BENCHMARK-CONTRACT.md`):

```python
import jrsuite
suite = jrsuite.Suite(key="MyBench", name="JR My Bench", version="1",
                      description="What it asks, in one sentence.",
                      columns=[jrsuite.Column("accuracy", "Accuracy", "Right answers"),
                               jrsuite.Column("time_s", "Time", "Wall time per run", unit="s")])
rows = [suite.row("my-model", "accuracy", 92.5, cases=40, machine="Rig"),
        suite.row("my-model", "time_s", 31.2, machine="Rig")]
bad = jrsuite.submit(jrsuite.connect(), rows)   # [] = every row recorded
```

- **`name`** is the tab; ours carry the `JR` prefix (Jon, 2026-10-11).
- **`version`**: bump it whenever a change could move a score. JR shows one version at a time.
- **Columns** in display order. `%` (default) is a pass rate: banded and counted in Overall.
  Any other unit is shown plain.
- **One model at a time, finished, or nothing.** A half-posted run reads like a finished one.
- **Setup fields** (`machine`, `quant`, `contextTokens`, `kvPrecision`, `thinking`, `engineName` ...)
  as measured, or omitted. A hosted model has no `machine`.
- **JR connection**: `JR_MCP_URL` and `JR_MCP_TOKEN`; here the openclaw `lab-dev` entry is the
  fallback.

## Cloud targets - the yardstick

`claude`, `claude-cli` and `chatgpt` are targets like any box; the answer shape is identical by
the time it leaves `ask_target()`.

- ⛔ **Keys live in the suite's `.env`** (gitignored, `chmod 600`), never in JR: `GET /api/settings`
  returns every setting in plaintext (Jon, 2026-08-28). A real environment variable wins.
- ⚠️ `claude` and `chatgpt` are metered. Prefer `claude-cli` (the subscription) for long suites; it
  cannot run tool cases.
- ⛔ **The `tool_use` id passes through verbatim** as the tool_call id; it is what lets a follow-up
  tool turn translate back without a side table.
