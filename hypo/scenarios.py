"""Synthetic incident scenario generator.

Produces scripted incidents in 5 categories. Each scenario bundles the logs, metrics and
deployment history that the mock tools serve, plus ground truth (root cause, required tools,
whether a rollback is appropriate).

Difficulty levels (per category, 6 test scenarios):
  easy (3)        clean signal, no decoy
  ambiguous (2)   a plausible decoy (e.g. an unrelated recent deploy) competes with the true cause
  adversarial (1) decoy + a misleading log line pointing at the decoy

`split="test"` uses alert openings 0-5 / clauses 0-1; `split="train"` uses openings 6-9 /
clauses 2-3 so a classifier trained on synthetic data never sees test phrasing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = ["bad_deployment", "database_slowdown", "memory_leak",
              "third_party_api_failure", "config_change"]

APP_SERVICES = ["checkout", "payments-api", "orders", "inventory", "search", "auth",
                "notifications", "recommendations", "shipping", "profile", "cart", "pricing"]

THIRD_PARTIES = [("PayGate", "api.paygate.example"), ("SMSHub", "api.smshub.example"),
                 ("GeoLookup", "api.geolookup.example"), ("TaxCalc", "api.taxcalc.example"),
                 ("MailRelay", "smtp.mailrelay.example"), ("FraudShield", "api.fraudshield.example")]

OPENINGS = [
    "ALERT [{svc}] error rate {pct}% (threshold 2%) sustained for 10 minutes.",
    "ALERT [{svc}] p99 latency {lat}ms and elevated 5xx responses. On-call paged.",
    "PagerDuty: {svc} SLO burn rate critical; success ratio dropped to {ok}%.",
    "Support escalation: customers report failures using {svc}; error budget is burning.",
    "ALERT [{svc}] availability {ok}% over the last 15 minutes; no infrastructure alarms firing.",
    "Synthetic check failing for {svc} from 3 regions; error rate {pct}%.",
    "SEV2 opened: degraded behaviour on {svc}, error rate {pct}%.",
    "Datadog monitor triggered on {svc}: 5xx ratio {pct}%, latency {lat}ms.",
    "Oncall page: {svc} unhealthy, success ratio {ok}%.",
    "Customer-facing errors from {svc}; dashboards show {pct}% failed requests.",
]

CLAUSES = {
    "bad_deployment": ["Errors began abruptly and affect every pod equally.",
                       "The failure appeared suddenly; traffic volume is unchanged.",
                       "All instances show the same failure pattern starting at one point in time.",
                       "Onset was sharp rather than gradual."],
    "database_slowdown": ["Requests are timing out and latency has been climbing.",
                          "Slowness grows over time and requests are waiting on something downstream.",
                          "Latency degrading progressively; timeouts rising.",
                          "Calls are hanging before failing."],
    "memory_leak": ["Pods have been restarting and performance has degraded gradually.",
                    "Instances are being recycled; the degradation built up over hours.",
                    "Restart count is increasing and responsiveness is worsening over time.",
                    "Containers keep getting killed and recovering."],
    "third_party_api_failure": ["Failures concentrate on requests that need outbound calls.",
                                "Internal dashboards look healthy but some features fail.",
                                "Only some request types fail; core resources look normal.",
                                "Calls leaving our network seem to be failing."],
    "config_change": ["Behaviour changed abruptly with no traffic change and no obvious code release.",
                      "The change in behaviour was sudden; resource usage looks normal.",
                      "Sudden shift in error pattern; hosts are otherwise healthy.",
                      "Something changed at one point in time but host metrics are fine."],
}

APP_METRICS = {  # name: (base, noise)
    "error_rate": (0.4, 0.1), "latency_p99": (180, 15), "cpu_usage": (35, 3),
    "memory_usage": (55, 1.5), "request_rate": (420, 20), "restarts": (0, 0),
    "db_pool_wait_ms": (4, 1), "dependency_latency": (90, 8), "upstream_5xx_rate": (0.1, 0.05),
}
DB_METRICS = {"db_query_latency_p99": (22, 3), "db_connections": (60, 4),
              "cpu_usage": (30, 3), "lock_waits": (1, 0.5)}

INFO_MSGS = ["GET /health 200 2ms", "request completed status=200 duration={d}ms",
             "cache hit ratio={r}", "metrics flushed", "worker heartbeat ok",
             "GC pause {g}ms (young)", "config poll: no changes", "TLS session resumed"]

BAD_DEPLOY_ERRORS = [
    "NullPointerException at OrderMapper.map(OrderMapper.java:88) build={v}",
    "KeyError: 'discount_code' in apply_promotions() build={v}",
    "TypeError: cannot read property 'currency' of undefined (PriceFormatter) build={v}",
    "JSON decode error: unexpected field 'items_v2' in request schema build={v}",
]
BAD_DEPLOY_SUMMARIES = ["Refactor order mapping and promotion handling", "Add multi-currency price formatting",
                        "Upgrade request schema to v2", "Optimise cart serialization"]
BENIGN_SUMMARIES = ["Update logging format", "Bump dependency versions", "Tweak dashboard labels",
                    "Reword error messages"]

DB_VARIANTS = [
    ("missing index on a table that outgrew its query plan",
     ["missing index", "seq scan", "index"],
     "slow query: SELECT * FROM orders WHERE customer_ref = ? took {x}ms (seq scan, rows=48M)"),
    ("long-running transaction causing lock contention",
     ["lock", "transaction", "batch"],
     "lock wait timeout exceeded; blocked by pid 4412 (batch-report job, idle in transaction {x}s)"),
    ("replica lag forcing reads to the primary and exhausting connections",
     ["replica", "lag", "connection"],
     "replica lag {x}s; failing reads over to primary; connection pool at max"),
]

LEAK_VARIANTS = [
    ("unbounded in-memory cache without eviction", ["cache", "eviction", "unbounded"],
     "cache 'session_cache' size={x} entries evictions=0"),
    ("file descriptor / connection handle leak", ["handle", "descriptor", "leak", "unclosed"],
     "open file descriptors={x} (limit 65536); unclosed HttpClient instances detected"),
    ("event listener leak", ["listener"],
     "registered event listeners={x} and growing; no deregistration on request end"),
]

CONFIG_VARIANTS = [
    ("db.pool.max_size", "50", "2", ["pool", "max_size"],
     "timeout acquiring connection from pool (max_size=2, waiting={x})"),
    ("http.client.timeout_ms", "5000", "50", ["timeout"],
     "outbound request exceeded http.client.timeout_ms=50 after {x}ms"),
    ("rate_limit.per_minute", "6000", "60", ["rate", "limit"],
     "429 Too Many Requests: rate_limit.per_minute=60 exceeded ({x} rejected)"),
    ("feature.new_pricing_engine", "false", "true", ["feature", "pricing"],
     "NewPricingEngine failed: missing tax table for region (feature flag enabled) x={x}"),
]


def _bump(v: str) -> str:
    a, b, _c = v[1:].split(".")
    return f"v{a}.{int(b) + 1}.0"


class Builder:
    def __init__(self, rng: random.Random, horizon: int, step: int):
        self.rng, self.horizon, self.step = rng, horizon, step
        self.tel: dict[str, dict] = {}

    def add_service(self, name: str, kind: str, queryable: bool = True):
        metrics_def = APP_METRICS if kind == "app" else DB_METRICS
        metrics = {m: self._series(base, noise, 0, "flat", base) for m, (base, noise) in metrics_def.items()}
        logs = []
        for _ in range(60):
            msg = self.rng.choice(INFO_MSGS).format(d=self.rng.randint(20, 120),
                                                    r=round(self.rng.uniform(.8, .97), 2),
                                                    g=self.rng.randint(5, 25))
            logs.append({"t": round(self.rng.uniform(0, self.horizon), 1), "level": "INFO", "msg": msg})
        self.tel[name] = {"kind": kind, "queryable": queryable, "logs": logs,
                          "metrics": metrics, "deployments": []}

    def _series(self, base, noise, onset, shape, peak):
        pts = []
        for t in range(self.horizon, -1, -self.step):
            v = base + self.rng.gauss(0, noise)
            if t <= onset:
                frac = (onset - t) / max(onset, 1)
                if shape == "step":
                    v = peak + self.rng.gauss(0, noise)
                elif shape == "ramp":
                    v = base + (peak - base) * frac + self.rng.gauss(0, noise)
            pts.append((t, round(max(v, 0), 2)))
        return pts

    def metric(self, svc, name, onset, shape, peak):
        base, noise = (APP_METRICS if self.tel[svc]["kind"] == "app" else DB_METRICS)[name]
        self.tel[svc]["metrics"][name] = self._series(base, noise, onset, shape, peak)

    def log(self, svc, level, msg, t):
        self.tel[svc]["logs"].append({"t": round(t, 1), "level": level, "msg": msg})

    def burst(self, svc, level, msg_pool, onset, count, **fmt):
        for _ in range(count):
            m = self.rng.choice(msg_pool) if isinstance(msg_pool, list) else msg_pool
            self.log(svc, level, m.format(x=self.rng.randint(120, 9000), **fmt), self.rng.uniform(0, onset))

    def deploy(self, svc, minutes_ago, version, previous, summary, dtype="deploy", extra=None):
        d = {"minutes_ago": minutes_ago, "type": dtype, "version": version,
             "previous_version": previous, "summary": summary, "author": self.rng.choice(["alice", "bob", "carol", "dev-bot"])}
        d.update(extra or {})
        self.tel[svc]["deployments"].append(d)


def _decoy_deploy(b, svc, minutes_ago):
    prev = f"v{b.rng.randint(1, 3)}.{b.rng.randint(0, 9)}.0"
    b.deploy(svc, minutes_ago, _bump(prev), prev, b.rng.choice(BENIGN_SUMMARIES))
    return _bump(prev)


def build_bad_deployment(b, svc, db, difficulty, idx):
    onset = b.rng.randint(15, 35)
    prev = f"v{b.rng.randint(1, 4)}.{b.rng.randint(0, 9)}.{b.rng.randint(0, 9)}"
    new = _bump(prev)
    b.deploy(svc, onset + 3, new, prev, b.rng.choice(BAD_DEPLOY_SUMMARIES))
    b.deploy(svc, b.rng.randint(2000, 3000), prev, "v0.9.0", "routine release")
    b.burst(svc, "ERROR", BAD_DEPLOY_ERRORS, onset, 14, v=new)
    b.metric(svc, "error_rate", onset, "step", b.rng.uniform(15, 40))
    b.metric(svc, "latency_p99", onset, "step", b.rng.uniform(300, 500))
    decoy = None
    if difficulty != "easy":
        decoy = "database_slowdown"
        b.burst(svc, "WARN", "slow query on " + db + " took {x}ms (below timeout)", 120, 5)
        b.metric(db, "db_query_latency_p99", 120, "step", 70)
    if difficulty == "adversarial":
        b.burst(svc, "ERROR", "connection reset by peer from " + db + " (retrying) x={x}", onset, 3)
    return dict(onset=onset, category="bad_deployment", detail=f"deployment {new} introduced a code bug",
                keywords=[new, "deploy", "release", "code"], rollback=True,
                rollback_version=prev, decoy=decoy, required=["get_deployments", "query_logs"],
                relevant=[svc], flavor="deploy")


def build_database_slowdown(b, svc, db, difficulty, idx):
    onset = b.rng.randint(20, 40)
    detail, kws, line = DB_VARIANTS[idx % len(DB_VARIANTS)]
    b.metric(db, "db_query_latency_p99", onset, "ramp", b.rng.uniform(2500, 6000))
    b.metric(db, "db_connections", onset, "ramp", 195)
    if "lock" in kws:
        b.metric(db, "lock_waits", onset, "ramp", b.rng.uniform(80, 200))
    b.burst(db, "WARN", line, onset, 8)
    b.metric(svc, "db_pool_wait_ms", onset, "ramp", b.rng.uniform(3000, 5000))
    b.metric(svc, "latency_p99", onset, "ramp", b.rng.uniform(4000, 7000))
    b.metric(svc, "error_rate", onset, "ramp", b.rng.uniform(10, 30))
    b.burst(svc, "ERROR", ["timeout acquiring connection from pool after 5000ms",
                           "query timeout after 5000ms on {db}"], onset, 14, db=db)
    pv = f"v{b.rng.randint(1, 4)}.{b.rng.randint(0, 9)}.0"
    b.deploy(svc, b.rng.randint(2000, 3000), pv, "v0.9.0", "routine release")
    decoy = None
    if difficulty != "easy":
        decoy = "bad_deployment"
        v = _decoy_deploy(b, svc, onset + b.rng.randint(30, 50))
        if difficulty == "adversarial":
            b.burst(svc, "WARN", "unknown field ignored in payload build=" + v + " x={x}", onset + 40, 3)
    return dict(onset=onset, category="database_slowdown", detail=detail, keywords=kws,
                rollback=False, rollback_version=None, decoy=decoy,
                required=["query_metrics", "query_logs"], relevant=[svc, db], flavor=detail)


def build_memory_leak(b, svc, db, difficulty, idx):
    onset = b.rng.randint(420, 560)  # slow ramp over hours
    detail, kws, line = LEAK_VARIANTS[idx % len(LEAK_VARIANTS)]
    from_deploy = idx % 2 == 0
    b.metric(svc, "memory_usage", onset, "ramp", 97)
    b.metric(svc, "restarts", 60, "ramp", b.rng.uniform(6, 12))
    b.metric(svc, "latency_p99", 90, "ramp", b.rng.uniform(1200, 2500))
    b.metric(svc, "error_rate", 30, "ramp", b.rng.uniform(5, 15))
    b.burst(svc, "WARN", line, onset, 8)
    b.burst(svc, "WARN", "full GC took {x}ms, heap 94%", 90, 6)
    b.burst(svc, "ERROR", ["java.lang.OutOfMemoryError: Java heap space",
                           "container OOMKilled (exit 137)"], 60, 8)
    prev = f"v{b.rng.randint(2, 4)}.{b.rng.randint(0, 9)}.0"
    rb_ver = None
    if from_deploy:
        new = _bump(prev)
        b.deploy(svc, onset + 20, new, prev, "Introduce in-process caching layer")
        rb_ver = prev
    else:
        b.deploy(svc, b.rng.randint(4300, 5000), prev, "v0.9.0", "routine release")
    decoy = None
    if difficulty != "easy":
        decoy = "third_party_api_failure" if idx % 2 else "database_slowdown"
        b.metric(svc, "request_rate", 90, "ramp", 560)  # mild traffic increase
        if difficulty == "adversarial":
            b.burst(svc, "WARN", "high request volume, shedding load x={x}", 90, 4)
    return dict(onset=onset, category="memory_leak", detail=detail, keywords=kws,
                rollback=from_deploy, rollback_version=rb_ver, decoy=decoy,
                required=["query_metrics"], relevant=[svc], flavor=detail)


def build_third_party(b, svc, db, difficulty, idx):
    onset = b.rng.randint(10, 30)
    name, host = THIRD_PARTIES[(idx + b.rng.randint(0, 5)) % len(THIRD_PARTIES)]
    b.burst(svc, "ERROR", ["upstream " + name + " returned 503 Service Unavailable",
                           "timeout after 5000ms calling " + host,
                           "circuit breaker OPEN for " + name], onset, 16)
    if difficulty == "easy":
        b.log(svc, "WARN", f"provider status: {name} reports partial outage", onset - 2)
    b.metric(svc, "error_rate", onset, "step", b.rng.uniform(12, 35))
    b.metric(svc, "latency_p99", onset, "step", b.rng.uniform(4500, 5200))
    b.metric(svc, "dependency_latency", onset, "step", 5000)
    b.metric(svc, "upstream_5xx_rate", onset, "step", b.rng.uniform(60, 95))
    pv = f"v{b.rng.randint(1, 4)}.{b.rng.randint(0, 9)}.0"
    b.deploy(svc, b.rng.randint(2000, 3000), pv, "v0.9.0", "routine release")
    decoy = None
    if difficulty != "easy":
        decoy = "bad_deployment"
        v = _decoy_deploy(b, svc, onset + b.rng.randint(8, 14))
        if difficulty == "adversarial":
            b.burst(svc, "WARN", "HTTP client retry policy changed build=" + v + " x={x}", onset + 10, 3)
    return dict(onset=onset, category="third_party_api_failure",
                detail=f"outage of external provider {name} ({host})", keywords=[name.lower(), host, "upstream", "provider"],
                rollback=False, rollback_version=None, decoy=decoy,
                required=["query_logs", "query_metrics"], relevant=[svc], flavor=name)


def build_config_change(b, svc, db, difficulty, idx):
    onset = b.rng.randint(15, 35)
    key, old, new, kws, line = CONFIG_VARIANTS[idx % len(CONFIG_VARIANTS)]
    n = b.rng.randint(20, 90)
    b.deploy(svc, onset + 2, f"cfg-{n}", f"cfg-{n - 1}", f"Config update: {key} {old} -> {new}",
             dtype="config", extra={"key": key, "old": old, "new": new})
    b.burst(svc, "ERROR", line, onset, 14)
    if difficulty != "adversarial":
        b.log(svc, "INFO", f"Config reload applied: {key} {old} -> {new}", onset + 1)
    b.metric(svc, "error_rate", onset, "step", b.rng.uniform(10, 30))
    if key == "db.pool.max_size":
        b.metric(svc, "db_pool_wait_ms", onset, "step", 2500)
        b.metric(svc, "latency_p99", onset, "step", 3000)
    decoy = None
    if difficulty != "easy":
        decoy = "database_slowdown" if key == "db.pool.max_size" else "bad_deployment"
        if decoy == "bad_deployment":
            _decoy_deploy(b, svc, onset + b.rng.randint(90, 130))
        else:
            b.metric(db, "db_query_latency_p99", 120, "step", 45)
    return dict(onset=onset, category="config_change", detail=f"config {key} changed {old} -> {new}",
                keywords=kws + [key], rollback=True, rollback_version=f"cfg-{n - 1}", decoy=decoy,
                required=["get_deployments", "query_logs"], relevant=[svc], flavor=key)


BUILDERS = {"bad_deployment": build_bad_deployment, "database_slowdown": build_database_slowdown,
            "memory_leak": build_memory_leak, "third_party_api_failure": build_third_party,
            "config_change": build_config_change}
DIFFICULTIES = ["easy", "easy", "easy", "ambiguous", "ambiguous", "adversarial"]


def generate_scenario(category: str, idx: int, seed: int, split: str = "test", difficulty: str | None = None,
                      sid: str | None = None) -> dict:
    rng = random.Random(f"{seed}-{category}-{idx}-{split}")
    difficulty = difficulty or DIFFICULTIES[idx % 6]
    svc = rng.choice(APP_SERVICES)
    db = f"{svc}-db"
    horizon, step = (720, 15) if category == "memory_leak" else (180, 5)
    b = Builder(rng, horizon, step)
    b.add_service(svc, "app")
    b.add_service(db, "db")
    gt = BUILDERS[category](b, svc, db, difficulty, idx)

    onset = gt["onset"]
    pct, lat = round(rng.uniform(8, 35), 1), rng.randint(900, 5200)
    op_idx = idx % 6 if split == "test" else 6 + rng.randint(0, 3)
    cl_idx = rng.randint(0, 1) if split == "test" else rng.randint(2, 3)
    alert = OPENINGS[op_idx].format(svc=svc, pct=pct, lat=lat, ok=round(100 - pct, 1)) + " " + CLAUSES[category][cl_idx]
    recent = sorted([l for l in b.tel[svc]["logs"] if l["level"] in ("ERROR", "WARN") and l["t"] <= 10],
                    key=lambda l: l["t"])[:3]
    snippet = [f"[t-{l['t']}m] {l['level']} {l['msg']}" for l in recent]

    return {
        "id": sid or f"INC-{CATEGORIES.index(category) * 6 + idx + 1:02d}",
        "split": split, "category": category, "difficulty": difficulty, "service": svc, "db_service": db,
        "alert": alert, "snippet": snippet, "telemetry": b.tel,
        "ground_truth": {"category": category, "detail": gt["detail"], "keywords": gt["keywords"],
                         "rollback_appropriate": gt["rollback"], "rollback_version": gt["rollback_version"],
                         "decoy_category": gt["decoy"], "required_tools": gt["required"],
                         "relevant_services": gt["relevant"], "onset_minutes_ago": onset},
    }


def generate_test_set(seed: int = 0) -> list[dict]:
    return [generate_scenario(c, i, seed) for c in CATEGORIES for i in range(6)]


def generate_train_set(n_per_category: int = 120, seed: int = 1000) -> list[dict]:
    out = []
    for c in CATEGORIES:
        for i in range(n_per_category):
            out.append(generate_scenario(c, i, seed, split="train", sid=f"TR-{c[:3]}-{i}",
                                         difficulty=DIFFICULTIES[i % 6]))
    return out


def save(scs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scs, indent=1))


def load(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/scenarios.json"
    scs = generate_test_set()
    save(scs, out)
    print(f"wrote {len(scs)} scenarios to {out}")
