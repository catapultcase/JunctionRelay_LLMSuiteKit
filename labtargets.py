"""labtargets - the lab's boxes and model adapters, shared by every suite here.

What a suite asks a model is the suite's business; HOW it reaches the model
(a llama-swap box, a Spark pair, the Anthropic API, the Claude Code CLI) and
what setup the running server reports are this module's. The boxes themselves
are the lab's: targets.json beside this file (gitignored; JR_LAB_TARGETS names
another path), shaped like targets.example.json.
"""
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

from datetime import datetime, timezone

# ⛔ A THINKING MODEL SPENDS THIS BEFORE IT ANSWERS. The default was 800, and on
# 2026-08-28 that scored qwen3.6-35b-a3b at 0% on every text tier: the
# transcripts show reasoning_content of 862, 846 and 801 tokens with content
# empty - the model was cut off mid-thought and never reached an answer. That is
# a measurement of the HARNESS, not the model, and it posted to the ledger as if
# it were the model.
#
# ⚠️ Some cases failed with reasoning of only 163-265 tokens and still no answer.
# THAT is the genuine fault - the model ends its turn still thinking - and it is
# the same one that silences Bishop. Raising the budget separates the two: what
# fails now fails on its own merit.
#
# ⛔ 4000 WAS STILL TOO LOW, AND STILL SCORED THE HARNESS (2026-08-28). A direct
# probe of qwen3.6-35b-a3b on the code_exec case returned finish_reason=length,
# completion_tokens=4000, 10387 characters of reasoning and 64 characters of
# answer - cut off mid-function. Every local model read 0% on code_exec and the
# number was about this constant. It was defended as a genuine stall on the
# strength of one transcript's character count, which is exactly the guess that
# finish_reason now exists to remove.
#
# A thinking model can spend thousands of tokens before it writes anything. The
# budget has to be big enough that stopping is the MODEL's choice; anything less
# and the suite measures its own ceiling.
#
# A case may still set its own max_tokens; this is only the default.
# ⛔ CLOUD TARGETS ONLY (2026-10-21). The lab's own boxes get no max_tokens at all, as
# OpenCode asks them: on the Spark pairs GPQA's hardest chemistry came back blank at
# 32000 and scored wrong, so the suite was measuring its ceiling again. See ask_target.
MAX_TOKENS = 32000

# ⛔ WHEN IS HITTING THE CEILING THE MODEL'S FAULT, NOT OURS?
#
# Below this budget it is OURS: the suite starved the model and any score is
# about the harness. That was true at 800, at 4000 and at 16000, and each time
# the fix was more budget, not a worse verdict for the model.
#
# At or above it, it is the MODEL'S: a level-5 MATH problem needs ~7,200
# completion tokens on qwen3.6-35b (measured), so 32,000 is over four times
# what the hardest question in the set requires. A case that exhausts it has
# not been cut off mid-thought, it has failed to terminate - and the clearest
# evidence is that one of the two cases to do so is a LEVEL 1 geometry
# question whose answer is 180 degrees.
#
# The threshold is explicit so that lowering MAX_TOKENS re-arms the protection
# automatically rather than silently blaming models for a budget cut.
GENEROUS_TOKENS = 32000

# The lab's boxes: {"<role>": {"api": "http://host:port", "machine": "<JR machine name>"}}.
# A container engine (vLLM, TensorFold) has no llama-swap and no /props; its setup is read by
# LLMSpeedTest's own capture (container_facts), the same fields the speed rows carry.
def _lab_targets():
    path = os.environ.get("JR_LAB_TARGETS") or os.path.join(HERE, "targets.json")
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return {}


TARGETS = {
    **_lab_targets(),

    # ── Cloud targets: the yardstick ─────────────────────────────────────────
    # ⛔ NO `machine`. A hosted model does not run on a box in the lab, and
    # inventing one would put a fact in the ledger that is not true. The
    # benchmark contract already makes machine optional for exactly this.
    #
    # ⚠️ Keys come from the ENVIRONMENT and are never read, logged or echoed by
    # this script: ANTHROPIC_API_KEY / OPENAI_API_KEY. A missing key fails the
    # run loudly rather than silently scoring zero.
    #
    # ⚠️ These cost real money per case, unlike the local boxes. The suite is
    # small (five tiers, a few dozen cases) but a cloud target is opt-in by
    # naming it, never included in --all.
    # ⚠️ METERED. This target spends API credit, and the account ran dry on
    # 2026-08-29 mid-run. Prefer "claude-cli" for anything long - MATH-500,
    # GPQA, RULER - which runs on the Claude Code subscription instead. Use
    # this one only when a case needs structured tool_calls, which a CLI
    # cannot produce.
    "claude": {
        "kind": "anthropic", "machine": None,
        "models": ["claude-opus-5"],
        "key_env": "ANTHROPIC_API_KEY",
    },
    # ⛔ Billed to the Claude Code subscription, not API credits. Runs where the
    # CLI is installed and signed in. No tools: see ask_cli.
    "claude-cli": {
        "kind": "cli", "machine": None, "cmd": "claude",
        "models": ["claude-opus-5"],
    },
    "chatgpt": {
        "kind": "openai", "api": "https://api.openai.com", "machine": None,
        "models": ["gpt-5.2"],
        "key_env": "OPENAI_API_KEY",
    },
}
SKIP_IDS = {"brain"}          # selectors are aliases, not models
TIMEOUT = 600
# A lab box answers uncapped, so a hard question can think for far longer than 600 s
# (GLM on Sparks A+B timed out on 13 GPQA questions, 2026-10-18). Two hours is only the
# backstop for a model that never stops; a timeout still blocks the post.
LAB_TIMEOUT = 7200


def container_facts(target, serving_id):
    """THE SETUP OF A CONTAINER ENGINE (vLLM, TensorFold on the Spark pairs), read exactly as LLMSpeedTest reads it
    for its speed rows - the launch argv, the engine's own log and probes - so a reasoning row and a speed row of one
    setup carry the same quant, KV, window, slots, MTP, thinking and engine (2026-10-10: Spark rows posted none).
    Uses the sibling JunctionRelay_LLMSpeedTest checkout (or JR_SPEEDTEST names it).
    {} when it is absent or the capture fails: absent is honest."""
    bench = _bench()
    if not bench:
        return {}
    try:
        bt = bench.TARGETS.get(target)
        if not bt:
            return {}
        cfg = bench.config_vllm(bt, serving_id)
    except Exception as e:
        print(f"    !! LLMSpeedTest capture failed ({e}) - the setup fields will post as null")
        return {}
    out = {"quant": cfg.get("quant"), "contextTokens": cfg.get("ctx"), "slots": cfg.get("parallel"),
           "kvPrecision": cfg.get("cache_k") or cfg.get("cache_v"),
           "mtp": bool(cfg.get("spec_type")) and cfg.get("spec_type") != "none",
           "thinking": cfg.get("thinking_default"), "vision": cfg.get("vision"),
           "engine": cfg.get("engine"), "engineName": cfg.get("engine_name"),
           "engineVersion": cfg.get("engine_version"), "sourceRepo": cfg.get("source_repo")}
    return {k: v for k, v in out.items() if v is not None}


def _bench():
    """LLMSpeedTest's bench module from the sibling checkout, or None."""
    sib = os.environ.get("JR_SPEEDTEST") or os.path.join(os.path.dirname(HERE), "JunctionRelay_LLMSpeedTest")
    if not os.path.isfile(os.path.join(sib, "bench.py")):
        return None
    if sib not in sys.path:
        sys.path.insert(0, sib)
    try:
        import bench
        return bench
    except Exception as e:
        print(f"    !! LLMSpeedTest import failed ({e})")
        return None


def llama_launch_facts(api, serving_id):
    """What /props cannot say for a llama-swap box, read from the RUNNING launch as LLMSpeedTest reads it: KV
    precision (-ctk/-ctv) and speculative decoding (--spec-type), so a reasoning row and a speed row of one setup
    carry the same fields (2026-10-10). Thinking is not inferred: a llama-server row says nothing until a flag
    proves it, as on the speed side. {} when the launch cannot be read."""
    bench = _bench()
    if not bench:
        return {}
    try:
        cmd = bench.running_cmd(api, serving_id) or ""
    except Exception:
        return {}
    out = {}
    m = re.search(r"-ctk\s+(\S+)", cmd) or re.search(r"-ctv\s+(\S+)", cmd)
    if m:
        out["kvPrecision"] = m.group(1)
    m = re.search(r"--spec-type\s+(\S+)", cmd)
    if m:
        out["mtp"] = m.group(1) != "none"
    return out


def runtime_facts(api, serving_id):
    """THE SETUP, AS THE RUNNING SERVER REPORTS IT (Jon, 2026-08-27: capture
    every field at run time).

    llama-server's /props describes what was ACTUALLY loaded: the file it
    opened, the slots it serves, the context it allocated, the build running.
    Everything else is a guess about it - the yaml says what we asked for, a
    filename says what someone named it, and the catalog can only say which
    quants we own. Absent is honest, so a field we cannot read is submitted as
    null and renders "-"; it is never inferred from a name.
    """
    try:
        with urllib.request.urlopen(f"{api}/upstream/{serving_id}/props", timeout=180) as r:
            d = json.loads(r.read().decode())
    except Exception:
        return {}
    out = {}
    base = str(d.get("model_path") or "").split("/")[-1]
    q = re.search(r"((?:UD-)?I?Q\d[\w_]*|BF16|F16|NVFP4)", base, re.I)
    if q:
        out["quant"] = q.group(1)
    if d.get("total_slots"):
        out["slots"] = int(d["total_slots"])
    ctx = (d.get("default_generation_settings") or {}).get("n_ctx")
    if ctx:
        out["contextTokens"] = int(ctx)
    if d.get("build_info"):
        out["engine"] = str(d["build_info"])
        # The runtime as data (JR 0.9.0.266): the pages show name + version and never parse engine.
        out["engineName"], out["engineVersion"] = "llama.cpp", str(d["build_info"]).split("-")[0]
    # The repo of the weights, when the path names one (an HF cache folder): JR links by it (0.9.0.272).
    m = re.search(r"models--([^/\\]+?)--([^/\\]+)", str(d.get("model_path") or ""))
    if m:
        out["sourceRepo"] = f"{m.group(1)}/{m.group(2)}"
    mods = d.get("modalities")
    if isinstance(mods, dict):
        out["vision"] = bool(mods.get("vision"))
    # ⚠️ MTP is the one field /props does NOT report, so it comes from the
    # serving id's '-mtp' suffix - the house naming convention, which is
    # authoritative for it. Every OTHER field is read from the server rather
    # than a name, because names were exactly what went wrong: a filename that
    # nothing verified is how a ledger ends up with no quant on any row.
    out["mtp"] = serving_id.endswith("-mtp")
    out.update(llama_launch_facts(api, serving_id))   # KV, and MTP from the launch itself when it says
    return out


# A serving id names a CONFIGURATION; the catalog holds MODELS WE HAVE
# DOWNLOADED. Quant, context window and MTP are ways of RUNNING a model, so they
# are stripped to find the catalog entry:
#
#     qwen3.8-27b-256k-q4-mtp  ->  qwen3.8-27b
#
# ⚠️ The quant strip is the part that is easy to leave out, and it fails silently
# when you do - the run measures fine and the results are simply never posted.
# The catalog collapsed quant-named entries into their models on 2026-08-27, so
# an id resolving to 'qwen3.8-27b-q4' matches nothing any more. The quant suffix
# must start with q/iq, which is what keeps it off names like 'lfm2.5-8b-a1b'
# and 'qwen3-embedding-0.6b'.
def catalog_name(serving_id):
    # A box that serves under an alias ('box-cd'): the model is whatever it is running, read from its launch.
    bench = _bench()
    running = bench.running_name(serving_id) if bench and hasattr(bench, "running_name") else None
    if running:
        return running
    s = serving_id.removesuffix("-mtp")
    s = re.sub(r"-\d+k(?=-|$)", "", s)
    return re.sub(r"-(?:iq|q)\d[a-z0-9_]*$", "", s, flags=re.I)


def resolve_model_id(ids, serving_id):
    """Catalog id for a serving id, trying progressively less specific names."""
    return (ids.get(serving_id)
            or ids.get(serving_id.removesuffix("-mtp"))
            or ids.get(re.sub(r"-\d+k(?=-|$)", "", serving_id.removesuffix("-mtp")))
            or ids.get(catalog_name(serving_id)))


def ask(api, model, messages, tools=None, max_tokens=MAX_TOKENS, auth=None, timeout=TIMEOUT):
    payload = {"model": model, "messages": messages, "stream": False}
    if max_tokens is not None:              # None = the server's own limit (the lab's boxes)
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    # A hosted OpenAI-compatible endpoint needs a bearer token; llama-swap does not.
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(f"{api}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        choice = json.loads(r.read().decode())["choices"][0]
    # ⛔ KEEP finish_reason. Without it an empty answer is ambiguous in exactly
    # the way that matters: "length" means WE starved the model and the score is
    # the harness's fault, "stop" means the model ended its turn having produced
    # nothing - a real and serious fault. Telling those apart by eye, counting
    # characters of reasoning against the token budget, is guesswork.
    msg = choice["message"]
    msg["finish_reason"] = choice.get("finish_reason")
    return msg


# ── Anthropic ────────────────────────────────────────────────────────────────
# Claude speaks a different shape from every other target here, so it is adapted
# at THIS ONE BOUNDARY and nowhere else: ask_anthropic returns the same
# {"content": str, "tool_calls": [...]} dict the OpenAI path returns, so every
# checker, every case file and the whole scoring path stay untouched.
#
# ⛔ The tool_use id is passed through VERBATIM as the OpenAI-style tool_call id.
# That is what lets a follow-up `tool` message be translated back into a
# tool_result block without keeping a side table of id mappings.
def _to_anthropic(messages, tools):
    """OpenAI-shaped conversation -> (system, messages, tools) for the Messages API."""
    system = "\n\n".join(m["content"] for m in messages
                          if m.get("role") == "system" and m.get("content"))
    conv = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            # A tool result is a USER turn carrying a tool_result block.
            conv.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or "call_0",
                "content": m.get("content") or "",
            }]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", "call_0"),
                               "name": fn.get("name", ""), "input": args})
            conv.append({"role": "assistant", "content": blocks})
        else:
            # ⛔ A message we built ourselves from a response may carry no role
            # at all. Passing None straight through produced
            # "messages.N.role: Input should be 'user', 'assistant'..." on
            # every second-turn case - the tool_calling and looping tiers -
            # which read as the MODEL failing when it was the adapter.
            conv.append({"role": role or "assistant", "content": m.get("content") or ""})

    # ⛔ EVERY tool_use MUST BE ANSWERED, IN ONE FOLLOWING USER TURN. Anthropic
    # rejects the request outright otherwise:
    #   "`tool_use` ids were found without `tool_result` blocks immediately after"
    # The OpenAI shape allows both things this forbids - a case answers only the
    # FIRST call when the model made several, and each result is its own `tool`
    # message - so a model that legitimately called two tools got a 400 and
    # scored as an ERROR. That read as the model failing when it was the
    # adapter, which is the worst kind of wrong number: it blames the subject.
    def _tool_use_ids(m):
        return [b["id"] for b in m["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use"] \
            if m["role"] == "assistant" and isinstance(m["content"], list) else []

    def _is_results_turn(m):
        return (m["role"] == "user" and isinstance(m["content"], list) and m["content"]
                and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"]))

    repaired, i = [], 0
    while i < len(conv):
        m = conv[i]
        repaired.append(m)
        ids = _tool_use_ids(m)
        if ids and i + 1 < len(conv):
            results, j = [], i + 1
            while j < len(conv) and _is_results_turn(conv[j]):
                results.extend(conv[j]["content"])
                j += 1
            have = {b["tool_use_id"] for b in results}
            for cid in ids:
                if cid not in have:
                    # The case declared one result; the others are genuinely
                    # unavailable. Say so rather than inventing an answer.
                    results.append({"type": "tool_result", "tool_use_id": cid,
                                    "content": "(the harness supplied no result for this call)"})
            order = {cid: n for n, cid in enumerate(ids)}
            results.sort(key=lambda b: order.get(b["tool_use_id"], 99))
            repaired.append({"role": "user", "content": results})
            i = j
            continue
        i += 1
    conv = repaired

    # OpenAI wraps the schema in a `function` object; Anthropic takes it flat.
    atools = [{
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
    } for t in (tools or []) if t.get("function")]
    return system, conv, atools


def ask_anthropic(model, messages, tools=None, max_tokens=MAX_TOKENS):
    import anthropic  # imported lazily: only a cloud run needs the SDK installed
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment
    system, conv, atools = _to_anthropic(messages, tools)
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": conv}
    if system:
        kwargs["system"] = system
    if atools:
        kwargs["tools"] = atools
    resp = client.messages.create(**kwargs)

    # ⛔ A REFUSAL IS NOT A WRONG ANSWER (Jon, 2026-08-28). This used to return
    # "[refused]" as the model's answer, which then failed the checker and
    # scored as a reasoning failure. It cost a real verdict: the 32K needle case
    # planted a "maintenance passphrase" and asked the model to print it, Claude
    # declined as it should, and the suite recorded long_context 0% - a safety
    # refusal reported as an inability to recall from long context.
    #
    # It is also incomparable by construction: a local model with no such
    # classifier is not being asked the same question. So the case did not run.
    if getattr(resp, "stop_reason", None) == "refusal":
        return {"role": "assistant", "content": "", "tool_calls": [],
                "finish_reason": "refusal"}

    text = "".join(b.text for b in resp.content if b.type == "text")
    calls = [{"id": b.id, "type": "function",
              "function": {"name": b.name, "arguments": json.dumps(b.input)}}
             for b in resp.content if b.type == "tool_use"]
    return {"role": "assistant", "content": text, "tool_calls": calls,
            "finish_reason": getattr(resp, "stop_reason", None)}


# ── Claude Code CLI ──────────────────────────────────────────────────────────
# ⛔ THE CLI IS NOT THE API, AND THAT IS THE POINT (Jon, 2026-08-28). A CLI run
# is billed to the Claude Code subscription, not metered API credits, and the
# expensive suites here - MATH-500, RULER - are long prompts run hundreds of
# times. Debugging this harness alone put ~200 opus calls on the API key.
#
# ⛔ IT CANNOT DO TOOLS, AND MUST NOT PRETEND TO. A CLI takes no per-case tool
# schema and returns no structured tool_calls, so a tool tier CANNOT run here.
# It raises, which scores ERROR (uncounted) and - under the all-or-nothing rule
# - stops the model posting at all. That is correct: a model measured on four
# tiers where others got five has not taken the same test.
def _cli_path(cmd):
    """⚠️ A ONE-SHOT `ssh host 'claude'` DOES NOT SEE ~/.local/bin. Ubuntu's
    .bashrc returns early for non-interactive shells, so the PATH that an
    interactive login has is not the PATH a scripted run gets - the same trap
    documented for devmachine's node. Resolve the binary ourselves."""
    import shutil, os.path
    return (shutil.which(cmd)
            or next((p for p in (os.path.expanduser(f"~/.local/bin/{cmd}"),
                                 f"/usr/local/bin/{cmd}") if os.path.exists(p)), cmd))


def ask_cli(model, messages, tools=None, max_tokens=None, cmd="claude"):
    import subprocess
    cmd = _cli_path(cmd)
    if tools:
        raise RuntimeError("a CLI target takes no tool schema and returns no "
                           "structured tool_calls - this tier cannot run here")
    parts = []
    for m in messages:
        role, text = m.get("role"), (m.get("content") or "")
        if not text:
            continue
        parts.append(text if role in ("system", "user")
                     else f"({role}) {text}")
    # ⛔ STRIP THE API KEY FROM THE CHILD'S ENVIRONMENT. load_dotenv() puts
    # ANTHROPIC_API_KEY in os.environ for the API targets, the CLI inherits it,
    # and then authenticates BY THAT KEY - which bills the metered API and
    # defeats the entire reason this target exists. It announced itself only as
    # "claude.ai connectors are disabled because ANTHROPIC_API_KEY..." on the
    # two calls that happened to fail; the ones that succeeded said nothing and
    # were billed anyway (Jon, 2026-08-29).
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    # ⚠️ STDIN, never argv: a RULER prompt at 64K tokens is ~256KB and would
    # run into the argument-length limit.
    out = subprocess.run([cmd, "-p", "--model", model],
                         input="\n\n".join(parts), capture_output=True,
                         text=True, timeout=TIMEOUT, env=env)
    # ⛔ A SAFEGUARDS REFUSAL ARRIVES AS EXIT 1, NOT AS A stop_reason. The CLI
    # has no structured refusal signal - it prints an API Error and exits
    # non-zero - so every suite's refusal handling, which checks
    # finish_reason, never fired for this target. GPQA discarded 58 good
    # answers over one declined question twice in a row for exactly this.
    detail_all = ((out.stdout or "") + (out.stderr or ""))
    if out.returncode != 0 and "safeguards flagged" in detail_all:
        return {"role": "assistant", "content": "", "tool_calls": [],
                "finish_reason": "refusal"}
    if out.returncode != 0:
        # ⛔ THE CLI PRINTS ITS API ERRORS TO STDOUT, NOT STDERR. Reporting only
        # stderr logged 32 failures as "exited 1: " with nothing after the
        # colon, and the actual cause - a safeguards refusal - was sitting in
        # stdout the whole time, unread (Jon, 2026-08-29).
        detail = ((out.stdout or "") + (out.stderr or "")).strip()
        raise RuntimeError(f"{cmd} exited {out.returncode}: {detail[:300] or '(no output)'}")
    return {"role": "assistant", "content": out.stdout.strip(), "tool_calls": [],
            "finish_reason": "end_turn"}


def ask_target(target, model, messages, tools=None, max_tokens=MAX_TOKENS):
    """One door for every backend. A local box and a hosted model differ in
    transport only; the answer shape is identical by the time it leaves here."""
    kind = target.get("kind", "openai_local")
    if kind == "anthropic":
        return ask_anthropic(model, messages, tools, max_tokens)
    if kind == "cli":
        return ask_cli(model, messages, tools, max_tokens, target.get("cmd", "claude"))
    key = os.environ.get(target["key_env"]) if target.get("key_env") else None
    if target.get("key_env") and not key:
        raise RuntimeError(f"{target['key_env']} is not set - export it before running this target")
    if kind == "openai_local":              # the lab's boxes: uncapped, like OpenCode
        return ask(target["api"], model, messages, tools, None, timeout=LAB_TIMEOUT)
    return ask(target["api"], model, messages, tools, max_tokens, auth=key)
