"""jrsuite - post a test suite's results to JunctionRelay's Benchmarks page.

ONE FILE, STANDARD LIBRARY ONLY. Copy it into your own suite; nothing else is
needed. JR keeps no list of suites: what its page shows about yours - the tab,
the blurb, the columns, their order, units and tooltips - is what you send here
(JunctionRelay docs/BENCHMARK-CONTRACT.md, "A suite describes itself").

    import jrsuite
    suite = jrsuite.Suite(
        key="MyBench", name="My Bench", version="1",
        description="What it asks, in a sentence a reader can act on.",
        columns=[jrsuite.Column("accuracy", "Accuracy", "Right answers, %"),
                 jrsuite.Column("time_s", "Time", "Wall time per run", unit="s")],
        source="MyBench run.py")
    rows = [suite.row("my-model", "accuracy", 92.5, cases=40, machine="Rig"),
            suite.row("my-model", "time_s", 31.2, machine="Rig")]
    jrsuite.submit(jrsuite.connect(), rows)     # all or nothing, confirmed

Where JR is: JR_MCP_URL and JR_MCP_TOKEN in the environment (or a .env beside
your script - see load_dotenv). In this lab the openclaw config's lab-dev entry
is the fallback, so there is one place the credentials live.
"""
import json
import os
import re
import urllib.request
from dataclasses import dataclass

OPENCLAW_CFG = os.path.expanduser("~/.openclaw/openclaw.json")


# ---------------------------------------------------------------- the suite
@dataclass
class Column:
    """One column of the suite's table.

    key          the metric suffix: a row's metric is "<family>_<key>"
    label        the heading (default: the key title-cased)
    description  the heading's tooltip
    unit         "%" (the default) is a pass rate: banded green/amber/red and
                 counted in Overall, weighted by cases. Anything else ("s",
                 "tokens", "count") is shown plain beside its unit and is NOT
                 banded or averaged in: 90 of an unknown unit is not "good".
    """
    key: str
    label: str = ""
    description: str = ""
    unit: str = "%"

    def declare(self):
        d = {"key": self.key}
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        if self.unit != "%":
            d["unit"] = self.unit
        return d


@dataclass
class Suite:
    """What the suite says about itself; every row carries it.

    key       stable id rows group on - never rename it lightly: JR groups by it
    name      the tab and heading
    version   bump whenever a change could move a score; JR never blends versions
    family    the metric prefix (default: the key lower-cased, letters and digits)
    display   "table" (the score is a property of the weights; machine is
              provenance) or "matrix" (the box is part of what was measured)
    """
    key: str
    name: str
    version: str
    description: str = ""
    columns: tuple = ()
    source: str = ""
    family: str = ""
    display: str = "table"

    def __post_init__(self):
        self.columns = tuple(self.columns)
        if not self.family:
            self.family = re.sub(r"[^a-z0-9]", "", self.key.lower())

    def config(self, **extra):
        """The configJson every row carries: the suite's self-description plus
        whatever run facts the caller adds (cases, failures, provider...)."""
        cfg = {"suite": self.key, "suite_name": self.name, "suite_version": self.version,
               "suite_columns": [c.declare() for c in self.columns]}
        if self.description:
            cfg["suite_description"] = self.description
        if self.display != "table":
            cfg["display"] = self.display
        cfg.update({k: v for k, v in extra.items() if v is not None})
        return cfg

    def row(self, model, column, value, *, cases=None, failures=None, notes=None,
            config=None, **fields):
        """One models_record_benchmark payload. `fields` are the row's setup
        (machine, quant, contextTokens, kvPrecision, mtp, thinking, engineName,
        engineVersion, sourceRepo, hardware ...); omit what you do not know -
        absent is honest, a guess is not. A hosted model has no machine."""
        if column not in {c.key for c in self.columns}:
            raise ValueError(f"{column!r} is not a declared column of {self.key}")
        cfg = self.config(cases=cases, failures=failures or None, **(config or {}))
        p = {"modelName": model, "metric": f"{self.family}_{column}", "value": value,
             "source": self.source or self.key, "configJson": json.dumps(cfg)}
        if notes:
            p["notes"] = notes
        p.update({k: v for k, v in fields.items() if v is not None})
        return p


# --------------------------------------------------------------- posting
def submit(mcp, rows):
    """Post ONE model's finished run, confirming every row from the server's own
    reply. Returns the failures; empty means every row landed.

    ⛔ ALL OR NOTHING, PER MODEL. Call this once a model has finished every case:
    a half-posted run reads exactly like a finished one and nothing marks it.
    ⛔ CONFIRMATION IS THE SERVER'S WORDS. models_record_benchmark can decline
    inside a successful response; a row counts only when the reply says
    "Recorded" (five rows were once reported posted and never written).
    """
    bad = []
    for p in rows:
        reply = (mcp.call("models_record_benchmark", p) or "").strip()
        if not reply.startswith("Recorded"):
            bad.append(f"{p.get('metric')}: {reply[:200] or '(empty reply)'}")
    return bad


def connect(client="jrsuite"):
    """A JR MCP session: JR_MCP_URL / JR_MCP_TOKEN, else the openclaw lab-dev entry."""
    url, token = os.environ.get("JR_MCP_URL"), os.environ.get("JR_MCP_TOKEN")
    if not url:
        try:
            s = json.load(open(OPENCLAW_CFG))["mcp"]["servers"]["lab-dev"]
            url, auth = s["url"], s["headers"]["Authorization"]
        except Exception:
            raise RuntimeError("JR not configured: set JR_MCP_URL and JR_MCP_TOKEN")
    else:
        auth = f"Bearer {token}" if token and not token.lower().startswith("bearer ") else (token or "")
    return JRMcp(url, auth, client)


class JRMcp:
    """Minimal MCP streamable-HTTP client."""

    def __init__(self, url, auth, client="jrsuite"):
        self.url, self.auth, self.sid, self._id = url, auth, None, 0
        r = self._post({"method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": client, "version": "1.0"}}})
        if "error" in r:
            raise RuntimeError(f"MCP initialize failed: {r['error']}")
        try:
            self._notify("notifications/initialized")
        except Exception:
            pass

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.auth:
            h["Authorization"] = self.auth
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        return h

    def _post(self, payload):
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, **payload}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self._headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            self.sid = r.headers.get("Mcp-Session-Id") or self.sid
            text, ctype = r.read().decode(), r.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in text.splitlines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == self._id:
                        return msg
            raise RuntimeError("MCP: no response event in SSE body")
        return json.loads(text) if text.strip() else {}

    def _notify(self, method):
        body = json.dumps({"jsonrpc": "2.0", "method": method}).encode()
        urllib.request.urlopen(
            urllib.request.Request(self.url, data=body, headers=self._headers()), timeout=30).read()

    def call(self, tool, args):
        r = self._post({"method": "tools/call", "params": {"name": tool, "arguments": args}})
        if "error" in r:
            raise RuntimeError(f"MCP {tool}: {r['error']}")
        result = r.get("result", {})
        text = "".join(c.get("text", "") for c in result.get("content", []))
        if result.get("isError"):
            raise RuntimeError(f"MCP {tool}: {text}")
        return text

    def model_ids(self):
        return {m.group(2): int(m.group(1))
                for m in re.finditer(r"\[(\d+)\]\s+(\S+)", self.call("models_query", {}))}


# ------------------------------------------------------------------ misc
def load_dotenv(path):
    """KEY=VALUE lines from `path` into the environment. A real environment
    variable always wins; values are never printed. Returns the NAMES set."""
    loaded = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
                    loaded.append(k)
    except FileNotFoundError:
        pass
    return loaded
