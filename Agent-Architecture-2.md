### Technical Note: Production-Grade AI PR Review Agent & Genesis Harness Architecture

---

#### Section 1: Overview & High-Level Architecture

Building production-grade AI systems requires moving beyond naive single-prompt setups toward **grounded multi-agent fan-out topologies** and **file-backed control loops**.

1. **AI PR Review Agent System:** Designed to reclaim senior developer attention by taking a GitHub PR webhook trigger, retrieving codebase context, fanning out across specialist reasoners, and publishing structured findings.


2. **Genesis Engineering Harness (`genesis-kit`):** An AI-native control harness used to construct and maintain the review agent. It offloads state management and governance from human memory into a persistent `.genesis/` disk spine.



```
               [ GITHUB PR WEBHOOK / TRIGGER ]
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │  INGRESS & ARQ QUEUE (Redis Checkpoints)  │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │     LANGGRAPH ORCHESTRATOR / FAN-OUT      │
        └──────┬──────────┬───────────┬──────────┬──┘
               │          │           │          │
               ▼          ▼           ▼          ▼
          ┌────────┐ ┌────────┐  ┌────────┐ ┌────────┐
          │Security│ │ Quality│  │ Tests  │ │ Docs   │  (Specialist Agents)
          └───┬────┘ └───┬────┘  └───┬────┘ └───┬────┘
              │          │           │          │
              └──────────┼───────────┴──────────┘
                         ▼
        ┌───────────────────────────────────────────┐
        │   RETRIEVAL PATH (Tiger Cloud Vector/     │
        │   TimescaleDB Agent Events Spine)         │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │      AGGREGATOR & CONFIDENCE GATE         │
        └──────┬─────────────────────────────┬──────┘
               │ High Confidence             │ Escalation / Low Confidence
               ▼                             ▼
   [ Auto-Post to GitHub ]      [ Human-in-the-Loop Approval Queue ]
                                             │
                                             ▼
                                ┌────────────────────────┐
                                │   L4 VERIFY (Genesis)  │
                                │ Separate Checker AI    │
                                └────────────────────────┘

```

---

#### Section 2: AI PR Review Agent Design Philosophy

##### Core Mission & First Principles

The system's goal is not to automate 100% of code review, but to **reclaim senior reviewer attention** by handling mechanical pattern recognition and surfacing only high-value, grounded findings.

##### Multi-Agent Fan-Out Topology

Instead of using a single LLM prompt, the orchestrator uses **LangGraph's Send API** to fan out across four specialist reasoners concurrently:

* **Security Specialist:** Analyzes vulnerabilities, data-flow safety, and injection risks.


* **Quality Specialist:** Checks code patterns, maintainability, and structural debt.


* **Tests Specialist:** Assesses test coverage, edge-case assertions, and test meaningfulness.


* **Docs Specialist:** Verifies API documentation, internal comments, and spec alignment.



##### Data Contracts & Persistence

* **Structured Findings Contract:** Every specialist outputs a standardized payload containing `agent_type`, `severity`, `category`, `file/line`, `confidence`, and `rationale`.


* **Unified Event Spine (Tiger Cloud / TimescaleDB Hypertable):** A single time-ordered table (`agent_events`) stores traces, tool calls, cost ledgers, and audit logs.


* **Durable Orchestration:** Worker queues use Redis and ARQ, while LangGraph checkpoints node execution state directly into Redis to handle worker failures gracefully.



##### Confidence-Weighted HITL Gate

Findings are passed to an Aggregator that computes an overall review score:

* **Auto-Post:** High-confidence reviews without severe bugs are automatically published back to GitHub.


* **Human Queue:** Low-confidence findings or any `CRITICAL` severity issues escalate to a Human-in-the-Loop (HITL) approval queue.



---

#### Section 3: Genesis Harness Architecture (`genesis-kit`)

##### Core Philosophy: "The Loop That Prompts Itself"

Genesis shifts control from interactive, manual turn-by-turn prompting to an **autonomous disk-backed execution loop**.

##### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      .genesis/ DISK SPINE                       │
│ ┌──────────────┬──────────────┬───────────────────────────────┐ │
│ │ PLAN.md      │ DONE.html    │ context-graph.json            │ │
│ │ (To-Do List) │ (Locked Spec)│ (Dependency Map & Invariants) │ │
│ └──────────────┴──────────────┴───────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FIVE-LOOP ENGINE                        │
│ ┌──────────┬──────────┬──────────┬──────────────┬─────────────┐ │
│ │ L1 BUILD │ L2 DEBUG │ L3 RES.  │ L4 VERIFY    │ G0 PRE-FLT  │ │
│ └──────────┴──────────┴──────────┴──────────────┴─────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MAKER ≠ CHECKER WALL                       │
│ [ Maker AI Executes Task ] ──► [ Checker AI Assesses Output ]   │
└─────────────────────────────────────────────────────────────────┘

```

1. **The `.genesis/` Disk Spine:**
* `PLAN.md` & `CURRENT.md`: Stores active work items, progress, and state across sessions.


* `DONE.html`: Contains locked definition-of-done specifications.


* `context-graph.json`: Maps system components, code dependencies, and immutable project invariants.




2. **The Five-Loop Engine:**
* **G0 Existence Pre-Flight:** Verifies existing code/assets before creating new modules to prevent duplicate implementations.


* **L1 Build Loop:** Executes lightweight code edits and automated test suites.


* **L2 Debug / L3 Research Loops:** Runs isolated diagnostics and information gathering when issues occur.


* **L4 Verify Loop:** The independent validation phase.




3. **Maker vs. Checker Separation:**
* Enforces strict isolation: the LLM that generates the code (**Maker**) is prohibited from approving its own work.


* A separate judge LLM (**Checker / L4 VERIFY**) evaluates the code against specifications without seeing the maker's intermediate reasoning transcript.





---

#### Section 4: Key System Integration Mechanics

| Layer | PR Review Agent Implementation | Genesis Harness Role |
| :--- | :--- | :--- |
| **Ingress & Queue** | GitHub Webhooks → Redis ARQ Queue → LangGraph Orchestration[cite: 1]. | Initializes `.genesis/` files via `G0`–`G6` setup routines. |
| **Execution** | Parallel specialist fan-out (Security, Quality, Tests, Docs)[cite: 1]. | Drives LLMs through five execution loops using local tools[cite: 2]. |
| **Verification** | Aggregator calculates confidence; routes `CRITICAL` findings to HITL queue[cite: 1]. | L4 independent Checker AI validates task outputs against `DONE.html`. |
| **Auditability** | Unified event spine log (`agent_events` hypertable)[cite: 1, 2]. | Tracks history via `implementation-notes.html` and `CURRENT.md`[cite: 1, 2]. |
