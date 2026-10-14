# Multi-Agent Orchestration: A Production Engineer’s Field Guide

> **Audience:** Experienced AI engineers building reliable, secure, and cost-controlled agentic systems.
>
> **Approach:** Build `hermes-lite` progressively: harness → agent loop → context → memory → orchestration → protocols → validation → security → cost → LLMOps → operations.
>
> **Core principle:** **Tools, skills, and deterministic boundaries matter more than agent count.** Multi-agent architecture primarily provides parallelism, specialization of permissions, and isolation—not automatic intelligence gains.

## Contents

1. [Foundations](#1-foundations)
2. [Harness and runtime](#2-harness-and-runtime)
3. [Agent loop and tools](#3-agent-loop-and-tools)
4. [Context engineering](#4-context-engineering)
5. [Memory](#5-memory)
6. [Multi-agent patterns](#6-multi-agent-patterns)
7. [MCP and A2A](#7-mcp-and-a2a)
8. [Deterministic validation](#8-deterministic-validation)
9. [Security](#9-security)
10. [Cost engineering](#10-cost-engineering)
11. [LLMOps](#11-llmops)
12. [Enterprise operations](#12-enterprise-operations)
13. [Testing and deployment](#13-testing-and-deployment)
14. [Capstone checklist](#14-capstone-checklist)
15. [Framework mapping](#15-framework-mapping)
16. [Further reading](#16-further-reading)

---

## 1. Foundations

### 1.1 Agent anatomy

```text
Request → Input normalization → Context assembly → Model decision
                                                ↓
Response ← Final validation ← Tool results ← Policy + tool execution
                              ↖── repeat as needed ──↗
        Cross-cutting: identity · budgets · tracing · security
```

The model is only one component. Reliability depends on the surrounding harness, tools, authorization, validation, and operational controls.

### 1.2 Five engineering principles

1. **Compounding error:** Under an independence assumption, 20 steps at 95% success each yield `0.95^20 ≈ 35.8%` end-to-end success; at 99%, `0.99^20 ≈ 81.8%`. Real failures are often correlated, so measure actual task success.
2. **Context is a budget:** Tokens consume latency, attention, and money; retrieval should be relevant and bounded.
3. **Models propose; code validates:** Enforce permissions and invariants outside the model, especially before irreversible actions.
4. **Tools before agents:** Improve tool contracts and task decomposition before adding agent handoffs.
5. **Trace every boundary:** Record versions, decisions, tool calls, outcomes, and cost with appropriate redaction.

### 1.3 When to add agents

| Situation | Default approach |
|---|---|
| Task and tools fit one coherent workflow | One agent |
| Independent subtasks can run concurrently | Orchestrator + parallel workers |
| Untrusted content needs restricted capabilities | Isolated reader / extraction worker |
| Different subtasks need different model sizes | Model routing; separate workers only if useful |
| Different tenants or compliance domains | Separate authorization and execution boundaries |
| “Specialists will debate and become smarter” | Benchmark first; often unnecessary |

### 1.4 Failure taxonomy

| Failure | Symptom | Primary mitigation |
|---|---|---|
| Non-termination | Repeated calls, runaway spend | Call caps, deadlines, repetition checks |
| Goal drift | Original objective disappears | Durable task state, bounded compaction |
| Tool misuse | Wrong tool or arguments | Orthogonal tools, typed schemas |
| Context rot | Long transcripts degrade performance | Retrieval, summaries, recent-turn window |
| False memory | Unsupported facts persist | Provenance, reviewable writes |
| Compounding errors | Plausible intermediate steps, bad outcome | Boundary-level validation |
| Prompt injection | Tool content becomes instruction | Least privilege, sandboxing, policy gates |
| Cost blowout | Excessive token/tool usage | Routing, caching, budgets |
| Regression | Previously working tasks fail | Traces, evals, staged releases |

---

## 2. Harness and runtime

### 2.1 Ephemeral execution

Each inbound event creates a **logical run** with isolated state and a deadline. A fresh process or container is an optional stronger isolation choice, not a requirement for every message.

```text
Gateway → authenticate + resolve tenant → dedupe → queue
                                                  ↓
                                      RunContext + state hydration
                                                  ↓
                                           agent loop
                                                  ↓
                                 persist result + audit + trace
                                                  ↓
                                           send response
```

Benefits: fault isolation, reproducibility, controlled credentials, and horizontal scaling. Costs: state hydration, process startup, and cache misses.

### 2.2 Canonical gateway contract

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class InboundMessage:
    platform: str
    external_id: str
    tenant_id: str
    user_id: str
    thread_id: str | None
    text: str

@dataclass(frozen=True)
class OutboundReply:
    tenant_id: str
    thread_id: str | None
    text: str
```

Authenticate **before** accepting tenant/user identifiers as authoritative. Scope deduplication to `(platform, tenant_id, external_id)` and use an atomic claim/lease; a simple `seen()` followed by `mark_seen()` can race. Acknowledge webhook requests promptly and process asynchronously.

### 2.3 Budgets and run state

```python
from dataclasses import dataclass, field
import time

@dataclass
class Budget:
    max_llm_calls: int = 40
    max_tool_calls: int = 30
    max_input_tokens: int = 300_000
    max_output_tokens: int = 40_000
    max_usd: float = 2.0
    wall_clock_s: float = 600.0

@dataclass
class RunContext:
    run_id: str
    session_id: str
    tenant_id: str
    user_id: str
    budget: Budget = field(default_factory=Budget)
    depth: int = 0
    deadline: float = field(default_factory=lambda: time.monotonic() + 600)
    parent_run_id: str | None = None
```

Budget accounting must be **shared or reserved atomically across parent and child runs**; scaling a copied budget does not enforce an aggregate spending ceiling. Check limits before expensive calls and charge actual usage afterward.

### 2.4 Versioned configuration

| Artifact | Purpose | Governance |
|---|---|---|
| `SOUL.md` | Identity and behavioral guidance | Reviewed and versioned |
| `skills/*.md` | Procedures and tool usage | Evaluated before promotion |
| `model.yaml` | Model routes and budgets | Reviewed and versioned |
| Database migrations | Durable state schema | Migration-controlled |
| Run manifest | Prompt hashes, model IDs, tool/config versions, commit SHA | Immutable audit reference |

**Implementation note:** Durable persistence, queue acknowledgment, and reply delivery require explicit failure handling. Use a transactional outbox where appropriate. Never mark an event completed or persist a nonexistent `result` in a `finally` block after a failed run.

---

## 3. Agent loop and tools

### 3.1 Loop contract

```text
assemble context
    ↓
check deadline and remaining budget
    ↓
LLM response ── final answer ──→ validate → return
    │
    └─ tool calls → validate schema → authorize → approve if needed
                                  ↓
                        execute with timeout
                                  ↓
                      append bounded tool result
                                  ↓
                              repeat
```

A production loop must handle **all** tool calls in a model response, preserve provider-specific tool-call/result correlation IDs, and record both successful and denied calls. Treat expected tool failures as structured results; propagate genuine infrastructure failures to the harness.

### 3.2 ReAct, planning, and hybrids

| Pattern | Best for | Trade-off |
|---|---|---|
| ReAct | Unpredictable exploration | Repeated reasoning calls |
| Plan → execute | Decomposable tasks | Stale plans |
| Plan → execute → conditional replan | Longer operational workflows | Extra control logic |

Start with one loop and a short, explicit task plan. Add separate planners only after tracing reveals a measurable benefit.

### 3.3 Tool design rules

1. Prefer a small set of distinct tools with precise names.
2. Use strict JSON Schema/Pydantic models and constrained decoding where supported.
3. Return structured, actionable errors without leaking sensitive internals.
4. Paginate and summarize large results; provide cursor handles.
5. Use **operation-specific** idempotency keys for mutations. A single `run_id` reused for different creates can incorrectly deduplicate legitimate actions.
6. Enforce deadlines, retries with jitter, and cancellation.
7. Classify capabilities as `read`, `write`, or `critical`; apply authorization to arguments, not just tool names.

```python
from typing import Literal
from pydantic import BaseModel

class ToolResult(BaseModel):
    ok: bool
    data: dict | None = None
    error: str | None = None
    repair_hint: str | None = None

class ToolSpec(BaseModel):
    name: str
    description: str
    danger: Literal["read", "write", "critical"]
    timeout_s: float = 20.0
```

### 3.4 Delegation

Send **goal + constraints + artifact handles + output schema**, not a raw parent transcript. Give each child a depth cap, restricted tool set, independent trace span, and reserved budget from the parent's remaining allocation.

```python
from pydantic import BaseModel, Field
from typing import Literal

class SubAgentReport(BaseModel):
    status: Literal["done", "partial", "failed"]
    findings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    spend_usd: float = 0.0
```

Validate child reports as **data**; schema validity alone does not establish factual correctness or safety.

### 3.5 End-loop guardrails

| Control | Initial policy |
|---|---|
| Model calls | 40 per run, tune from telemetry |
| Tool calls | 30 per run |
| Deadline | 10 minutes, workload-dependent |
| Repeated calls | Detect identical signatures and non-progress cycles |
| Cost | Tenant-level and run-level hard limits |
| Final answer | Typed result with `done`, `partial`, or `failed` |
| Degradation | Stop new exploration near budget exhaustion |

### 3.6 Scheduled runs

Treat cron as a synthetic inbound event with stricter capabilities and budgets. Give each scheduled job a stable idempotency key and reviewable destination. Never let unattended jobs silently promote self-authored skills or perform unapproved critical mutations.

---

## 4. Context engineering

### 4.1 Prompt assembly

```text
Stable: system policy → selected skills → tool contracts
Variable: authorized memory → session summary → scratchpad
Recent: bounded transcript → current objective
```

Keep stable provider-cacheable prefixes byte-identical where practical. Cache economics depend on provider, model, token thresholds, and retention policy; **do not assume fixed discount multipliers**. Measure hit rate and actual billed input.

### 4.2 Compaction

At a measured context threshold (for example, 70% of the available window): preserve system/tool policy, extract decisions and open tasks into structured scratchpad state, summarize older turns, retain recent turns, and record the compaction event in traces. Do not treat a summary as authoritative over source records.

```python
class TaskState(BaseModel):
    objective: str
    decisions: list[str] = Field(default_factory=list)
    pending_actions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
```

### 4.3 Retrieval choices

Start with lexical search (FTS/BM25) for identifiers, error codes, and tickets. Add vector or hybrid retrieval when evaluations show paraphrase recall failures. SQLite FTS5 is a good single-node starting point; PostgreSQL full-text search is a natural shared-database option; `pgvector` adds semantic retrieval without an extra service.

---

## 5. Memory

| Tier | Stores | Lifecycle |
|---|---|---|
| Procedural | Skills/runbooks | Draft → eval → review → release |
| Semantic | Facts with provenance | Propose → verify → supersede/expire |
| Episodic | Events, transcripts, outcomes | Append → summarize → retain/delete |

### 5.1 Episodic recall

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_episodes_scope
    ON episodes (tenant_id, user_id, session_id);
```

For SQLite FTS5, maintain an external-content index and join matches back to the authoritative `episodes` table. Enforce `(tenant_id, user_id)` authorization on **every** query; a tenant filter alone can expose one user's memories to another.

### 5.2 Skills

```yaml
---
name: postgres-triage
version: 3
status: verified
trigger: "slow queries or database performance"
preconditions:
  - "read-only PostgreSQL credential"
budget_hint:
  max_tool_calls: 15
---
```

A skill body can prescribe collecting `pg_stat_statements`, inspecting plans, and reporting findings. `EXPLAIN ANALYZE` **executes the query** and can cause side effects for statements/functions with side effects; use safe `EXPLAIN` and database-enforced read-only permissions by default.

### 5.3 Typed facts

```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_event_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    superseded_at TEXT
);
```

Store provenance, consent where relevant, confidence, and supersession history. Conflicting facts should be flagged or reconciled using source authority—not automatically replaced solely because the subject and predicate match.

### 5.4 Consolidation

Use a scheduled, idempotent process to summarize events and propose facts/skills. Route sensitive or behavior-changing proposals through verification and review. Memory writes must honor retention, deletion, and tenant boundaries.

---

## 6. Multi-agent patterns

| Pattern | Best use | Primary risk |
|---|---|---|
| Orchestrator → workers | Independent parallel subtasks | Context-poor briefs |
| Hierarchical planner | Deep decomposition | Goal drift |
| Fixed pipeline | Known stages and typed handoffs | Unnecessary model calls |
| Shared-state graph | Resumable, stateful workflows | State/transition complexity |
| Generator + critic | Difficult quality-sensitive tasks | Cost and correlated errors |

**Operational rules:** cap depth; reserve shared budgets; pass artifacts rather than transcripts; validate every handoff; propagate trace context; cancel orphaned workers; enforce each worker's own permissions.

The **context-baton problem** is the loss of assumptions and evidence at each handoff. Use a brief template containing objective, inputs, constraints, source handles, success criteria, and expected schema. If delegation overhead exceeds worker execution cost, reconsider the split.

---

## 7. MCP and A2A

### 7.1 Decision rule

| Boundary | Default |
|---|---|
| Agent ↔ tools/resources | MCP |
| Agent ↔ independent agent service | A2A |
| Two internal functions in one process | Direct calls |

### 7.2 MCP

MCP uses a host/client/server design and exposes **tools**, **resources**, and **prompts**. Typical transports include local `stdio` and remote Streamable HTTP. Treat server-provided schemas, descriptions, and results as untrusted input; validate, classify, authorize, and restrict credentials independently.

```python
# Illustrative MCP server; supply your own ticket repository.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("corp-tools")

@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """Return an authorized ticket summary for a ticket ID."""
    return {"id": ticket_id, "status": "example"}

if __name__ == "__main__":
    mcp.run()
```

**Bridge responsibilities:** namespace tools by server, enforce per-tool authorization, constrain outputs, set timeouts, validate result schemas, and monitor tool-list changes. MCP discovery is **not** an authorization decision.

### 7.3 A2A

A2A provides agent discovery through an **Agent Card**, task/message/artifact exchange, and asynchronous task lifecycles. A task can move through states such as submitted, working, input-required, completed, failed, or canceled; exact state names and transport bindings depend on the pinned specification/SDK version.

```text
Client agent → discover card → authenticate → submit task
             → poll/stream updates → input-required? → human response
             → collect artifacts → validate and authorize consumption
```

Do not pass another agent's prose directly into privileged tool instructions. Apply bounded polling, cancellation, idempotency, task ownership checks, and artifact validation. Prefer a pinned SDK and contract tests over copying pseudocode tied to rapidly changing APIs.

---

## 8. Deterministic validation

### 8.1 Before tool execution

1. Parse and validate arguments against the tool schema.
2. Resolve the caller's identity and permission for the specific resource.
3. Check budgets, deadline, and danger class.
4. Require human approval or a revocable grant for high-risk actions.
5. Execute under least-privilege credentials with a timeout.

### 8.2 Read-only SQL: defense in depth

A SQL AST parser is useful for rejecting obviously unsafe statements, but **parsing alone cannot guarantee read-only execution**: CTEs, functions, permissions, and dialect quirks matter. Use a database role with `SELECT`-only grants, transaction read-only mode, statement timeout, row limits, and an allowlisted schema. Prefer approved query templates for sensitive data.

### 8.3 Post-execution checks

Bound tool-output size, redact sensitive data, verify domain invariants, and use deterministic arithmetic. For financial totals use `Decimal`, explicit rounding rules, and the correct tax base.

```python
from decimal import Decimal

def invoice_total(lines: list[Decimal], tax_rate: Decimal) -> Decimal:
    subtotal = sum(lines, Decimal("0"))
    return (subtotal * (Decimal("1") + tax_rate)).quantize(Decimal("0.01"))
```

### 8.4 Final answers

Return a typed envelope: `status`, `reply`, `evidence`, `artifacts`, `warnings`. If validation fails, retry with bounded repair hints; otherwise return `partial` or escalate. Never turn a failed validation into a false success.

### 8.5 Mutations

Use **prepare → preview → approve → commit**. Bind approval to the exact normalized action and parameters, set an expiry, re-check authorization at commit, and use an operation-specific idempotency key. A dry run is informative but is not a database transaction or a guarantee that external state will remain unchanged.

---

## 9. Security

### 9.1 Prompt-injection threat model

Untrusted documents, pages, emails, MCP outputs, tool descriptions, and stored memory may contain instructions intended to redirect the agent. The core failure is promoting **data into authority**. Prompt-only defenses are insufficient.

### 9.2 Layered controls

1. **Mark untrusted content** and preserve provenance; do not mistake delimiters for a security boundary.
2. **Separate capabilities:** untrusted readers should lack secrets and privileged write tools.
3. **Validate handoffs:** accept constrained claims/evidence, not instructions to the next agent.
4. **Enforce egress at the network layer:** domain and destination restrictions, DNS/IP checks, proxy logging.
5. **Authorize every action:** per-user, per-tenant, per-resource policy and approval for critical operations.
6. **Review persistent memory and skill updates:** no untrusted content should silently become policy.
7. **Red-team regressions:** test malicious tool outputs, canary leaks, and attempted unauthorized actions.

### 9.3 Reader isolation

```text
Untrusted source → restricted reader (no secrets, limited egress)
                 → structured claims + source references
                 → validator / policy gate
                 → privileged agent (still verifies before acting)
```

Isolation reduces blast radius but **does not make extracted claims trustworthy**. The privileged side must independently check facts and policy.

### 9.4 Canary controls

Canary strings and honeypot tools can provide detection signals, not prevention guarantees. Avoid placing realistic secrets or personal data into prompts merely to create traps; secure egress, permissions, and auditing remain primary.

---

## 10. Cost engineering

**Optimize cost per successful task**, not raw cost per token.

### 10.1 Metering

Track model/provider, input/output/cache tokens, tool latency, retries, child runs, total cost, and task outcome. Meter at run and tenant level, and reconcile estimates against provider invoices.

### 10.2 Optimization levers

| Lever | Why it helps | Caveat |
|---|---|---|
| Stable-prefix caching | Avoids repeatedly billing full shared context | Provider/model-specific terms |
| Model routing | Cheap models for extraction and summarization | Validate quality by task class |
| Retrieval + pagination | Avoids enormous tool results | Requires cursor lifecycle |
| Compaction | Bounds long-running context | Can discard critical evidence |
| Parallelism only when useful | Reduces wall-clock time | Often raises total cost |
| Offline batch inference | Lower-cost noninteractive workloads | Delayed results; availability varies |
| Sampled evaluation | Controls judge spend | Rare failures may be missed |

### 10.3 Budget states

| State | Behavior |
|---|---|
| Under 80% | Normal work |
| 80–90% | Stop speculative retrieval, prioritize deliverable |
| Over 90% | Produce partial result with evidence and pending actions |
| Hard limit | Abort safely and report what is complete |

### 10.4 Illustrative cost calculation

Suppose a run has 60,000 input tokens, of which 85% are cached at **10% of the normal input price** (hypothetical), plus 2,000 output tokens. At $3/M input and $15/M output:

```text
Uncached input = 60,000 × 0.15 = 9,000
Cached input   = 60,000 × 0.85 = 51,000
Input cost     = (9,000 + 51,000 × 0.10) / 1,000,000 × $3 = $0.0423
Output cost    = 2,000 / 1,000,000 × $15 = $0.0300
Total          = $0.0723 (excluding tools, retries, and overhead)
```

This is a **scenario**, not a current provider price quote. Real savings depend on the exact cache policy and token usage.

---

## 11. LLMOps

### 11.1 Trace → evaluate → diagnose → gate → release

Use OpenTelemetry-compatible spans for runs, model calls, tools, and child tasks. GenAI semantic conventions evolve; pin the version and avoid assuming every attribute name is stable. Redact prompts, secrets, and user data before export where policy requires.

### 11.2 Evaluation tiers

| Tier | Scope | Gate |
|---|---|---|
| Unit | Validators, policy, budget accounting | Every commit |
| Mocked loop | Scripted model/tool trajectories | Every commit |
| Golden tasks | Representative real-model tasks | Pre-merge or pre-release |
| Online | Sampled production outcomes | Continuous monitoring |
| Adversarial | Injection, exfiltration, privilege escalation | Release blocking |

LLM judges need anchored rubrics, human calibration, and disagreement analysis. A different model family can help reduce some correlated errors but does not guarantee an unbiased verdict.

### 11.3 CI gate example

```yaml
id: ticket-triage-01
tags: [triage, security]
input:
  fixture: fixtures/tickets/001.md
checks:
  - type: forbidden_tool_calls
    tools: [send_email, run_sql_write]
  - type: final_schema_valid
  - type: no_canary_leak
  - type: judge_rubric
    rubric: rubrics/triage.yaml
    min_score: 3
```

Version prompts, skills, model routes, policy, tool schemas, and evaluation datasets. Release by shadowing and canaries; keep rollback manifests ready. Exact model-output replay is not always possible even with identical inputs and a seed—preserve recorded trajectories for deterministic harness replay.

---

## 12. Enterprise operations

### 12.1 Multi-tenancy and identity

Scope memory, artifacts, traces, budgets, and concurrency by tenant and user. Derive identity from verified authentication—not caller-supplied strings. Use delegated user tokens where the agent acts on behalf of a person, and scoped service identities for scheduled work.

### 12.2 Data governance

Classify sensitive inputs, define retention per store, enforce residency where applicable, and implement deletion across raw events, summaries, facts, embeddings, derived artifacts, and eligible trace data. Do not put credentials into model context.

### 12.3 Human approval

Approval requests should include the proposed action, exact arguments, preview/diff, risk class, requester, expiration, and audit links. Timeout means no action—not implicit approval. Record approvals and denials as audit and evaluation signals.

### 12.4 Compliance and audit

Capture actor, model/config manifest, authorization decision, tool parameters (redacted), approvals, and outcomes. Regulatory classification depends on jurisdiction and **specific use case**, not merely on whether a system is an “agent”; obtain legal/compliance review for employment, credit, healthcare, and other regulated decisions.

### 12.5 Reliability

Track task success, latency percentiles, error rates, budget aborts, cost/task, approval backlog, and injection alerts. Add circuit breakers and a fleet-wide read-only kill switch checked at each tool boundary. Incident sequence: restrict capabilities → revoke affected credentials → preserve evidence → diagnose → fix → evaluate → staged rollout.

---

## 13. Testing and deployment

### 13.1 Test pyramid

1. Deterministic unit tests for schemas, policy, budgets, and arithmetic.
2. Mocked-model loop tests with recorded trajectories and failures.
3. Small real-model golden-task integration suite.
4. Larger statistical evaluation suite.
5. Chaos tests: timeouts, rate limits, duplicate events, worker crashes.
6. Security tests: injection, privilege escalation, exfiltration, memory poisoning.

### 13.2 Deployment options

| Topology | Use case |
|---|---|
| Single service/container | Development and low-volume pilot |
| Gateway + queue + worker pool | Default scalable production design |
| Private inference endpoints | Data-residency/security requirements |
| Dedicated tenant pools | Strong isolation requirements |

Use queue visibility/lease extension or heartbeats for long tasks, a dead-letter queue for poison messages, and idempotent handlers. Separate memory consolidation and offline evaluation from the request path.

---

## 14. Capstone checklist

### 14.1 Reference architecture

```text
                     ┌────── LLMOps control plane ──────┐
                     │ traces → evals → gates → rollout │
                     └────────────────┬─────────────────┘
Gateway → auth → dedupe → queue → run harness + budgets
                                     │
                         context + memory retrieval
                                     │
                         model decision / loop
                                     │
                 schema → policy → approval → execution
                                     │
                  tools (MCP) / agents (A2A) / workers
                                     │
                       result validation + persistence
                                     │
                                  response

Cross-cutting: tenant isolation · provenance · cost · audit · kill switch
```

### 14.2 Maturity model

| Level | Required capabilities |
|---|---|
| **0 — Demo** | Working loop and basic tools |
| **1 — Reliable** | Typed tools, budgets, deadlines, traces, manifests |
| **2 — Operable** | Evals, cost/task, memory lifecycle, approval workflow |
| **3 — Hardened** | Isolation, per-tool auth, egress controls, injection tests, kill switch |
| **4 — Enterprise** | Tenant governance, residency, auditability, staged releases, incident SLOs |

### 14.3 Recommended build order

1. **Build:** Sections 1–5 (single agent, tools, context, memory).
2. **Harden:** Sections 8–10 (validation, security, cost).
3. **Operate:** Section 11 (tracing and eval gates).
4. **Extend:** Sections 6–7 (workers, MCP, A2A as justified).
5. **Ship:** Sections 12–13 (governance, testing, deployment).

---

## 15. Framework mapping

| Concept | LangGraph | OpenAI Agents SDK | Claude Agent SDK |
|---|---|---|---|
| Control flow | Nodes/edges and graphs | Runner and agents | Built-in agent loop |
| Delegation | Subgraphs / nodes | Handoffs / agents as tools | Subagents |
| Durable state | Checkpointers and store | Sessions / application state | Application-managed state/files |
| Policy | Custom nodes and middleware | Guardrails and application checks | Hooks and permissions |
| Human approval | Interrupt/resume | Application HITL mechanisms | Permission flows/hooks |
| MCP | Integration support | Integration support | Integration support |

**Framework selection:** choose the smallest abstraction that supports your persistence, approval, and observability needs. Framework features and APIs change; verify against the version you deploy.

---

## 16. Further reading

- **Model Context Protocol specification:** `https://modelcontextprotocol.io/specification`
- **Agent2Agent project:** `https://a2a-protocol.org/`
- **OpenTelemetry GenAI conventions:** `https://opentelemetry.io/docs/specs/semconv/gen-ai/`
- **Anthropic — Building Effective Agents:** `https://www.anthropic.com/engineering/building-effective-agents`
- **Simon Willison — Prompt injection:** `https://simonwillison.net/tags/prompt-injection/`
- **Hermes Agent repository:** `https://github.com/NousResearch/hermes-agent`
- **Langfuse documentation:** `https://langfuse.com/docs`

> **Scope note:** Code snippets are educational contracts and examples, not a complete runnable repository. Integrations, authentication, transaction boundaries, SDK imports, and provider-specific behavior must be implemented and tested before deployment.
