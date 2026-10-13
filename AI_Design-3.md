# Whiteboard-Style System Design Prompts (RAG / Agentic AI / LLM Production)

These are open-ended, verbal-practice prompts — not MCQs. In a real interview you'd talk through these at a whiteboard in 8–12 minutes. For each: think out loud, state assumptions, cover the trade-offs, then check the "strong answer covers" section — don't peek first. Aim to say your constraints and trade-offs *explicitly*, since that's what's actually being scored, not just landing on "a RAG system with re-ranking."

A useful spoken structure for all of these: **clarify constraints → high-level architecture → walk through the critical path → call out 2-3 trade-offs and why you chose one side → mention how you'd validate and monitor it.** Interviewers are listening for that shape as much as the specific answer.

---

## Prompt 1: Customer Support RAG at Scale

*"Design a RAG-based customer support assistant for an e-commerce company with 2 million help-center articles and product docs, expecting 50,000 queries/day, with a requirement that answers must never contradict official policy."*

**Think about:** chunking strategy for policy documents specifically, why hybrid retrieval matters here (order numbers, SKUs), how you'd prevent contradiction of official policy (constrained generation vs. citation-forcing vs. a verification pass), and what happens when no relevant document exists.

<details>
<summary>Strong answer covers</summary>

- **Chunking**: semantic chunking by section/policy clause rather than fixed token windows, since policy documents have meaningful boundaries and mid-clause splits could cause a technically-true-but-misleading partial answer.
- **Hybrid retrieval**: dense + BM25, since order numbers/SKUs/product codes need exact-match, not just semantic similarity.
- **Two-stage retrieval**: fast hybrid first pass (top 50) → cross-encoder re-rank (top 3-5) for precision, given the corpus size.
- **Contradiction prevention**: strict system prompt requiring citation of retrieved chunk IDs for every factual claim, ideally with a post-generation verification step (does the answer's claims map back to cited chunks?) rather than trusting the model to self-police.
- **No-answer case**: explicit fallback ("I don't have information on this, let me connect you to a human agent") rather than letting the model guess — critical for a policy-sensitive domain where hallucination has real cost.
- **Freshness**: incremental reindexing as policies change, with versioning so you can tell *which* policy version an answer was grounded in (audit trail matters for support/compliance).
- **Validation**: golden eval set built specifically from known policy edge cases, not just general Q&A pairs.
</details>

---

## Prompt 2: Multi-Hop Research Agent

*"Design an agent that answers questions requiring synthesis across multiple internal documents, e.g., 'What was the impact of our Q2 pricing change on churn in the enterprise segment, and how does that compare to the SMB segment?'"*

**Think about:** why single-pass RAG fails here, how you'd decompose the question, how the agent decides it has "enough" evidence to stop, and how you'd prevent it from silently dropping one part of a multi-part question.

<details>
<summary>Strong answer covers</summary>

- **Why single-pass fails**: one query embedding can't simultaneously retrieve "Q2 pricing change," "enterprise churn," and "SMB churn" data well — these are effectively 3+ distinct retrieval targets bundled in one question.
- **Query decomposition**: break the compound question into sub-questions (pricing change details; enterprise churn data; SMB churn data) either via an explicit planning step or the ReAct pattern (reason → retrieve → observe → reason again).
- **Iterative/multi-hop retrieval**: retrieve for sub-question 1, use that context to inform retrieval for sub-question 2, etc. — rather than one flat retrieval call.
- **Stopping condition**: the agent needs an explicit check — "have I gathered evidence for every clause of the original question?" — otherwise it's prone to answering the easiest sub-part and silently dropping the comparison the user actually asked for.
- **Verification/self-critique step**: before finalizing, re-read the original question against the draft answer to check all parts were addressed — a lightweight critic pass.
- **Failure mode to flag explicitly**: partial-answer silent drop (answering churn impact but skipping the segment comparison) is the single most common failure in compound queries, and worth naming unprompted.
</details>

---

## Prompt 3: Real-Time Latency-Constrained RAG

*"Design a RAG system for a voice assistant where end-to-end response latency must stay under 800ms, including retrieval and generation."*

**Think about:** where the time budget actually goes, what you're willing to sacrifice for speed, and how re-ranking/multi-hop retrieval (both discussed as "best practice" elsewhere) might need to be cut or restructured here.

<details>
<summary>Strong answer covers</summary>

- **Budget the 800ms explicitly**: e.g., ~50-100ms retrieval, ~500-600ms generation (dominant cost), leaving headroom for network/orchestration — stating a rough budget out loud shows systems thinking.
- **Cut cross-encoder re-ranking or make it optional**: it adds meaningful latency for marginal precision gain; at this budget, a good hybrid first-pass retrieval alone, tuned for precision, is likely the trade-off — call this out explicitly as *sacrificing* some precision for latency, not pretending there's no cost.
- **Streaming generation**: stream tokens as generated so perceived latency (time-to-first-token) is far lower than full-response latency — important distinction for voice specifically, where the user needs to hear something quickly even if the full answer takes longer.
- **Smaller/distilled model or model routing**: a smaller model for the generation step trades some quality for the latency budget; routing complex queries to a slower path with a caveat, and keeping the common case fast.
- **Avoid multi-hop retrieval** for this use case — no time budget for iterative retrieve-reason loops; the design should explicitly state that multi-hop is a bad fit here, unlike Prompt 2.
- **Pre-computation**: cache common queries/responses, precompute embeddings, warm up model instances to avoid cold-start latency spikes.
</details>

---

## Prompt 4: Agent with High-Stakes Tool Actions

*"Design an agentic system that can take actions on a user's behalf — cancel orders, issue refunds, modify subscriptions — using natural language requests."*

**Think about:** what's fundamentally different about risk here vs. a Q&A agent, where human-in-the-loop fits, how you'd prevent the agent from hallucinating a tool call or wrong arguments, and how you'd handle an ambiguous request ("cancel my stuff").

<details>
<summary>Strong answer covers</summary>

- **Risk tiering**: not all actions are equal — read-only actions (check order status) can be fully autonomous; low-risk reversible actions (pause subscription) can be autonomous with logging; high-risk/hard-to-reverse actions (refund, cancellation) require explicit user confirmation or human-in-the-loop approval before execution.
- **Structured tool calls with schema validation**: the model outputs a structured function call (tool name + typed arguments); the orchestration layer validates against a strict schema before execution — reject and re-prompt on malformed/hallucinated tool calls rather than executing them.
- **Confirmation step for destructive actions**: before executing a refund/cancellation, the agent should state back what it's about to do and get explicit confirmation ("You want to cancel order #4471 for $89.99 — confirm?") — this catches both agent misunderstanding and ambiguous user intent.
- **Ambiguity handling**: "cancel my stuff" is underspecified — the agent should ask a clarifying question (which order? which subscription?) rather than guessing and acting, especially given the action is destructive.
- **Audit logging**: every action taken needs a full trace (what was requested, what tool was called, what arguments, what confirmation was given) for compliance and post-incident debugging.
- **Guardrails against prompt injection**: since this agent takes real-world actions, injected instructions (e.g., from a malicious support ticket the agent reads) are higher-stakes here than in a pure Q&A system — worth explicitly naming this as a bigger deal than in Prompt 1's system.
</details>

---

## Prompt 5: Debugging a Live Incident

*"Your RAG-based agent has been in production for 3 months. Overnight, customer complaints spike: the assistant is giving answers that reference outdated pricing that changed last week. Walk me through your live debugging process."*

**Think about:** this is a "narrate your triage" prompt — the interviewer wants to hear a systematic process, not just the final answer ("the index is stale"). Talk through what you'd check first, second, third, and why in that order.

<details>
<summary>Strong answer covers</summary>

- **Reproduce first**: confirm the issue with a direct test query about the pricing change before assuming root cause — rules out a one-off user error report.
- **Isolate the stage**: check retrieval first — is the *old* pricing document still being retrieved (index staleness) or is the *new* pricing document retrieved but the model still cites the old figure (generation/faithfulness failure, or a caching issue upstream)?
- **Most likely root cause given the symptom**: outdated pricing pointing specifically to *last week's* change strongly suggests an indexing/freshness problem (new doc wasn't ingested, or old doc wasn't invalidated) rather than a model behavior problem — say this hypothesis out loud and how you'd confirm it (check ingestion pipeline logs/timestamps for that specific document).
- **Check for caching**: a semantic or exact-match response cache could be serving a stale answer even if the index is fine and updated — worth checking as a parallel hypothesis, not just the index.
- **Immediate mitigation vs. root-cause fix**: while diagnosing, consider an immediate mitigation (invalidate/force-refresh the specific stale document or clear the relevant cache entries) separate from the underlying pipeline fix (why didn't the incremental update pick up the change automatically) — interviewers want to see you distinguish "stop the bleeding" from "fix root cause."
- **Post-incident**: propose a monitoring addition (alert on documents older than X days since last verified update, or a scheduled freshness check) so this class of failure is caught automatically next time rather than via user complaints.
</details>

---

## Prompt 6: Evaluating a RAG System You Didn't Build

*"You've just joined a team with an existing RAG system in production. There's no formal evaluation process — decisions on prompt/model changes are made by 'does it look better in a few manual tests.' How do you introduce rigor without halting shipping velocity?"*

**Think about:** this tests judgment/pragmatism as much as technical knowledge — a good answer is incremental, not "stop everything and build a perfect eval framework first."

<details>
<summary>Strong answer covers</summary>

- **Start with a small golden set**: pull 30-50 real representative queries (ideally from actual user logs, covering common + edge cases) with human-verified correct answers/expected retrieved chunks — this is achievable quickly and immediately better than nothing.
- **Component-level metrics from day one**: track Recall@k for retrieval and a faithfulness/groundedness check for generation separately, not just end-to-end "looks good," so future debugging can isolate stage of failure.
- **Regression gate, not full framework**: require any prompt/model/retrieval change to run against the golden set before merging — lightweight CI-style gate, not a large upfront investment.
- **Introduce online signal incrementally**: add thumbs up/down or explicit feedback capture if not present, since it's low-effort and gives real production signal over time to expand and correct the golden set.
- **Pragmatic framing**: explicitly acknowledge the trade-off — you're not blocking shipping velocity, you're adding a safety net proportional to risk, starting small and expanding coverage over time rather than demanding a mature framework upfront.
</details>

---

## How to Practice This Round

Read one prompt, set a timer for 8–10 minutes, and talk out loud (even alone) through: constraints → architecture → critical path → trade-offs → validation/monitoring. Then check the answer. The goal isn't matching every bullet — it's noticing if you skipped an entire dimension (e.g., you designed the architecture but never mentioned how you'd validate it, or never stated a trade-off explicitly). That omission pattern is usually the actual gap, more than any single fact.

