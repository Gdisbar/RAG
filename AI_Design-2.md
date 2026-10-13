# System Design, Validation, Deployment & Debugging (RAG / Agentic AI / LLM Production)

This set skips definitional questions and focuses on scenario-based reasoning — the kind that separates "knows the terms" from "has actually run this in production." Each question presents a system state or symptom; you diagnose or design against it. Difficulty: Medium–Difficult, weighted toward Difficult.

---

## Part 1: Concept Review — The Senior-Level Layer

### 1. System Design Patterns for RAG at Scale

**Two-stage retrieval architecture**: a production system almost never does single-pass dense retrieval. Standard pattern is (1) cheap, high-recall first pass — dense ANN search or hybrid dense+BM25 over the full index, pulling top 50–100 candidates, then (2) expensive, high-precision re-ranking (cross-encoder or LLM-based) down to the top 3–5 that actually go into the prompt. Designing "retrieve top-5 directly with a cross-encoder" is a common wrong answer — it doesn't scale to large corpora.

**Index freshness architecture**: for frequently-updated corpora (docs, tickets, code), you need an ingestion pipeline with incremental updates (upsert changed docs only, not full reindex) plus a way to invalidate/version stale embeddings when the source doc changes. A full reindex on every update is the naive answer that doesn't scale; versioned/incremental indexing with a document-hash or last-modified check is the production answer.

**Routing/model cascades**: senior system design often expects a router step — classify the query first (simple factual lookup vs. complex multi-step reasoning vs. needs a tool) and dispatch to different pipelines (cheap small model + RAG vs. full agent loop). This controls cost/latency at scale instead of running every query through the most expensive path.

**Caching layers**: semantic caching (cache based on embedding similarity of the query, not exact string match) catches near-duplicate questions ("what's your refund policy" vs "how do refunds work") that exact-match caching misses.

**Context window budget management**: with a fixed token budget, a senior design question is *how do you allocate it* — system prompt, retrieved chunks, conversation history, and the user query all compete. Common answer: dynamically truncate/summarize conversation history first (least information-dense) before cutting retrieved context (most directly relevant to the current answer).

### 2. Validation & Evaluation at the System Level

**Golden dataset + regression testing**: before any prompt, model, or retrieval change ships, senior teams run it against a fixed "golden" eval set with known correct answers, tracking metrics over time — this catches silent regressions (e.g., a new re-ranker improves average recall but breaks a specific query category).

**Offline vs. online evaluation**: offline (golden set, human-labeled) tells you if a change is *directionally* safe before shipping; online (A/B test, live traffic, real user feedback like thumbs up/down) tells you actual impact, because production query distribution rarely matches your offline eval set exactly — a frequent trap question is "your offline eval improved, why did production get worse," and the answer usually points to eval-set/production distribution mismatch.

**Component-level vs. end-to-end evaluation**: debugging a bad RAG answer requires isolating *where* it failed — was retrieval bad (check Recall@k on that specific query), or was retrieval fine but generation ignored the context (check faithfulness)? Evaluating only the end-to-end answer quality hides which stage to fix.

**Shadow deployment / canary testing**: before fully switching to a new model or pipeline version, run it in parallel on live traffic without serving its output (shadow) or serve it to a small % of traffic (canary) to catch regressions before full rollout — a direct extension of the "blue-green rollback" concept from your original report, but for LLM/RAG systems specifically.

**Human-in-the-loop eval and feedback loops**: thumbs up/down or explicit correction from users can be fed back as a continuously growing eval set — but a known pitfall is feedback bias (only unhappy users click "thumbs down," or a UI making one option easier than another skews the data).

### 3. Deployment & Production Reliability

**Rollback strategy for LLM/RAG systems**: unlike a simple blue-green rollback for a stable model, RAG systems have *multiple independently-versioned components* — the index, the retrieval config, the prompt template, and the LLM itself. A senior-level answer distinguishes rolling back the whole system vs. rolling back just one component (e.g., reverting a bad prompt change without re-indexing).

**Latency budget under load**: production systems must account for tail latency (p95/p99), not just average latency — a re-ranking step or multi-hop retrieval adds average latency that might be fine at p50 but blow the SLA at p99 under load. Senior design questions often ask you to identify *which stage* is the tail-latency risk (usually generation/decoding length or a synchronous tool call, not embedding lookup).

**Graceful degradation**: if the vector DB, a tool, or an external API is down or slow, a production agent/RAG system should degrade gracefully (e.g., fall back to a cached answer, a simpler non-RAG response, or a clear "I can't access that right now" rather than silently hallucinating or hanging).

**Cost-aware autoscaling**: LLM inference cost scales with both request volume and token count per request (input + output). A senior design answer distinguishes scaling *compute* (more GPU replicas for volume) from scaling *token budget management* (shorter contexts, model routing) — throwing more replicas at a token-length cost problem doesn't fix it.

### 4. Debugging Playbook (the "diagnose this trace" skill)

For any bad agent/RAG output, the systematic debugging order senior candidates are expected to know:
1. **Reproduce and isolate**: is it retrieval, generation, or orchestration/tool-calling?
2. **Check retrieval first** (if RAG): were the right chunks retrieved at all? (Recall@k check)
3. **Check grounding**: if right chunks were retrieved, did the model actually use them? (faithfulness check)
4. **Check tool/agent trace** (if agentic): did the model call the right tool, with correct arguments, and correctly interpret the tool's output?
5. **Check for compounding errors**: in multi-step agent traces, an error at step 2 can look like a "reasoning failure" at step 5 — trace backward to the first wrong step, not just the final wrong output.
6. **Check for non-determinism**: is the failure consistent (a real bug) or intermittent (sampling/temperature variance, race condition in tool calls, flaky external API)?

A common wrong answer pattern in "debug this" MCQs: jumping straight to "the model needs fine-tuning" when the actual root cause is retrieval, a prompt instruction gap, or a tool schema mismatch — all fixable without touching model weights.

---

## Part 2: Practice MCQs (Scenario-Based, Medium–Difficult)

**Q1.** You're designing a RAG system over a corpus of 10 million documents. A colleague proposes running the cross-encoder re-ranker directly over the full corpus for every query to maximize precision. What's the main problem with this design, and what's the standard fix?
A) Cross-encoders can't be used with large corpora at all — remove re-ranking entirely
B) It's computationally prohibitive at that scale; use a cheap first-pass retriever (dense/hybrid ANN) to get top-k candidates, then re-rank only those
C) Cross-encoders require the corpus to be under 1 million documents by design
D) Switch to a larger embedding model instead of re-ranking

**Q2.** Your offline eval shows a new retrieval config improves Recall@5 by 8%, so you ship it. A week later, production user satisfaction scores drop. What's the most likely explanation?
A) The eval metric Recall@5 is fundamentally invalid
B) Offline golden-set query distribution doesn't match live production query distribution, so gains on the eval set didn't transfer
C) Users dislike higher recall
D) The vector database silently corrupted data

**Q3.** A corpus of internal documents is updated hourly (new tickets, edited docs). Your current design does a full reindex of all embeddings every hour. What's the main problem as the corpus grows, and the better design?
A) There's no problem; full reindexing is always correct
B) Full reindexing cost grows with total corpus size regardless of how few docs changed; incremental upsert of only changed/new documents (tracked via hash or last-modified timestamp) scales independent of total corpus size
C) Switch to reindexing daily instead of hourly
D) Use a smaller embedding model to make full reindexing cheaper

**Q4.** A user reports that the RAG system's answer to their multi-part question is factually wrong. Your debugging trace shows the correct chunks were retrieved (Recall@k = 1.0 for this query) but the final answer contradicts them. Where should you look next, and what's this failure category called?
A) Look at the embedding model — this is a retrieval failure
B) Look at the prompt/generation step — this is a faithfulness/groundedness failure, since retrieval already succeeded
C) Look at the chunking strategy — this is a chunk-boundary failure
D) Look at the vector database index — this is a staleness failure

**Q5.** You're deploying a new prompt template for your RAG system. What deployment strategy best catches regressions before they affect most users?
A) Deploy directly to 100% of traffic and monitor error logs afterward
B) Canary deployment: route a small percentage of live traffic to the new template while comparing key metrics against the current version before full rollout
C) Only test on the offline golden dataset, then deploy to 100%
D) Ask a single internal user to try it once

**Q6.** In a multi-agent pipeline (planner → researcher → coder → critic), the final output is wrong. The critic agent approved it. Where should debugging start?
A) Immediately fine-tune the critic agent's underlying model
B) Trace backward through the agent sequence to find the first step where the output diverged from correct, since later agents may have compounded or failed to catch an earlier agent's error
C) Assume the final coder agent is always the point of failure
D) Restart the entire pipeline with a higher temperature

**Q7.** Your production LLM system's p50 latency is well within SLA, but p99 latency frequently breaches it under load. Which is the most likely cause to investigate first?
A) Average token count per request, since it drives p50 too
B) A synchronous tool call or a variable-length generation step (e.g., long-tail long responses or a slow external API call) that only some requests hit, disproportionately affecting tail latency
C) The embedding lookup step, since it's always the slowest part
D) The system prompt length, since it's identical for every request

**Q8.** Your vector database becomes temporarily unavailable in production. What is the best graceful-degradation strategy for the RAG-based chatbot?
A) Let the LLM generate an answer without any retrieved context and present it as normal
B) Fall back to a cached response or a clear message indicating degraded capability, rather than silently generating an ungrounded (and potentially hallucinated) answer
C) Retry the vector DB call indefinitely until it responds, blocking the user
D) Switch to a larger LLM to compensate for missing context

**Q9.** Your team wants to reduce LLM inference cost that has grown 3x over two months with no increase in request volume. Adding more GPU replicas hasn't helped cost-per-request. What's the more likely root cause and correct lever?
A) More replicas should always fix cost — the autoscaling config is broken
B) Average tokens per request (input context or output length) has likely grown; the fix is managing token budget (shorter context, prompt compression, model routing for simple queries) rather than scaling compute
C) The embedding model needs to be upgraded
D) Switch cloud providers

**Q10.** You need to route "what's my order status" (simple lookup) and "compare these three vendor contracts and flag risks" (complex multi-step reasoning) through the same system. What's the senior-level design choice?
A) Run every query through the most capable agent pipeline for consistency
B) Add a lightweight query classifier/router that dispatches simple queries to a cheap direct-lookup or small-model+RAG path, and complex queries to the full agent/reasoning pipeline
C) Only support one query type to keep the system simple
D) Increase the context window so both query types use the same static prompt

**Q11.** A semantic cache (caches responses keyed by embedding similarity of the query) is proposed to reduce redundant LLM calls. What's the main risk this introduces that an exact-match string cache doesn't have?
A) It can't cache anything at all
B) Two queries that are similar in embedding space but have meaningfully different intent (e.g., "cancel my subscription" vs "pause my subscription") could incorrectly return the same cached answer
C) It only works for numerical data
D) It requires no similarity threshold tuning

**Q12.** During a rollback discussion, a teammate proposes reverting the entire RAG system (index, retrieval config, prompt, and model) because the prompt template update caused a regression. What's the senior-level critique of this plan?
A) Full rollback is always correct and simplest
B) Since RAG systems have independently-versioned components, only the specific component that regressed (the prompt template) needs to be rolled back — reverting the index or model unnecessarily risks losing unrelated improvements
C) You should never roll back a prompt template
D) The index and model can't be versioned separately from the prompt

**Q13.** Your context window budget is tight: system prompt + retrieved chunks + conversation history + user query don't all fit. What's the generally preferred order to trim first, and why?
A) Trim retrieved chunks first, since they're the least important
B) Trim conversation history first (often least information-dense/most redundant), preserving system prompt and retrieved chunks, which are most directly relevant to answering the current query
C) Trim the system prompt first since it's static
D) Trim the user's current query first to save the most space

**Q14.** An agent intermittently fails a task — sometimes it succeeds, sometimes it doesn't, with the same input. What should you check before concluding there's a logic bug in the agent's reasoning?
A) Immediately rewrite the entire agent prompt
B) Check for non-determinism sources: sampling temperature/randomness in generation, race conditions in parallel tool calls, or flakiness in an external API/tool the agent depends on
C) Assume the model is incapable of the task and switch models
D) Increase the number of agent steps allowed

**Q15.** A stakeholder asks how you'd validate a new RAG pipeline before full production rollout, beyond just "check if it feels better." What's the most complete senior-level answer?
A) Only run it against the offline golden eval set, then ship
B) Run offline regression testing against a golden dataset to catch known-answer regressions, then shadow/canary deploy on live traffic to validate real-world performance and gather online metrics (e.g., user feedback, latency) before full rollout
C) Ship to 100% of users and monitor complaints
D) Have one engineer manually test 5 queries

---

## Answer Key & Explanations

**Q1: B.** Cross-encoders score query-document pairs jointly and don't scale to full-corpus search — the standard architecture always uses a cheap high-recall first pass (dense/hybrid) to narrow candidates, then re-ranks only the top-k.

**Q2: B.** This is the classic offline/online evaluation gap: golden-set improvements don't guarantee production gains if the eval set's query distribution doesn't represent what real users actually ask — a core reason online A/B testing matters even after good offline results.

**Q3: B.** Full reindexing cost scales with total corpus size, making it increasingly wasteful and slow as the corpus grows, when only a small fraction of documents actually changed. Incremental/versioned indexing (upsert only changed docs, tracked by hash or timestamp) decouples update cost from total corpus size.

**Q4: B.** Correct retrieval (Recall@k=1.0) rules out retrieval/chunking/staleness as the cause — the failure is downstream, at generation, where the model didn't faithfully ground its answer in the retrieved context.

**Q5: B.** Canary deployment lets you compare the new template's real-world performance against the current version on a small slice of live traffic, catching regressions before they impact all users — stronger than offline-only testing or a full blind rollout.

**Q6: B.** In multi-agent pipelines, errors compound — a later agent (like the critic) approving a bad output doesn't mean it's the source of the error. Senior debugging traces backward to find the first divergence point rather than assuming the last or most visible agent is at fault.

**Q7: B.** p50 being fine while p99 breaches under load points to a subset of requests hitting a variable-cost path — typically a synchronous external call or unusually long generation — rather than a cost that's incurred uniformly by every request (which would show up in p50 too).

**Q8: B.** Silently generating an answer without retrieved context risks confident hallucination presented as normal — the safer production pattern is degrading visibly (cached response or explicit "degraded" messaging) rather than invisibly.

**Q9: B.** If request volume is flat but cost tripled, and more compute didn't reduce per-request cost, the token count per request (context growth, longer outputs) is the more likely driver — this needs token-budget management, not just more replicas.

**Q10: B.** A router/classifier dispatching by query complexity is the standard cost/latency control at scale — sending every query through the most expensive pipeline wastes resources on simple lookups.

**Q11: B.** Semantic similarity doesn't guarantee same intent — near-neighbor embeddings can represent meaningfully different (sometimes opposite) requests, so returning a cached answer for a "similar" but different-intent query risks wrong answers, a risk exact-match caching doesn't have.

**Q12: B.** Because RAG components (index, retrieval config, prompt, model) are independently versioned, isolating the rollback to the actually-regressed component avoids unnecessarily discarding unrelated improvements in other components.

**Q13: B.** Conversation history is often the most redundant/least critical to the *current* answer compared to the system prompt (defines behavior) and retrieved chunks (directly relevant facts), making it the usual first candidate for truncation or summarization under tight token budgets.

**Q14: B.** Intermittent failure on identical input is a signature of non-determinism — sampling randomness, race conditions, or flaky dependencies — not necessarily a deterministic logic bug, and should be ruled out before concluding the reasoning itself is broken.

**Q15: B.** The complete validation pipeline combines offline regression testing (catches known regressions safely, before touching real users) with shadow/canary deployment (validates real-world performance on live traffic) before full rollout — neither alone is sufficient at senior-level rigor.

---

## A Note on Interview Framing

At senior level, interviewers (and tools like Zeero) often phrase these as "here's a symptom, what's the root cause and fix" rather than "define X." The strongest signal you can give in a live interview is stating your debugging *order* explicitly (reproduce → isolate retrieval vs. generation vs. orchestration → check for compounding/non-determinism) before jumping to a fix — this is exactly the reasoning pattern tested above.