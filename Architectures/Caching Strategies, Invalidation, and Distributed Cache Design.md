# Caching Strategies, Invalidation, and Distributed Cache Design

> **Audience:** Senior engineers, staff architects, and technical leads  
> **Purpose:** A practical architecture reference for choosing cache patterns, controlling staleness, preventing cache-driven outages, selecting eviction policies, and scaling caches across nodes and regions.

---

## 1. Executive Summary

Caching places a less-durable representation of data closer to the consumer to reduce access latency and load on the source of truth. It may store raw rows or derived projections such as rendered fragments, aggregates, search results, or serialized API responses.

The central design question is:

> **After a write reaches the source of truth, when does each cache learn about it?**

That question determines the consistency window, latency, complexity, and failure modes of the design.

Key principles:

1. A cache is a deliberate consistency trade-off—not free performance.
2. Cache-aside is a useful default, but read-through and several write patterns solve different workloads.
3. Expiration can synchronize traffic and create a stampede; TTL jitter alone does not protect a single hot key.
4. LRU can be polluted by one-time scans. Eviction policy should match the access pattern.
5. Cache penetration, breakdown, and avalanche are different failure modes and need different defenses.
6. Event-driven invalidation can reduce stale windows, but event delivery, ordering, and idempotency become part of correctness.
7. Distributed and multi-level caches require explicit routing and invalidation designs.
8. Cache failures must not cascade into database outages: observe misses and latency, and use rate limits, circuit breakers, and bulkheads.

## 2. Caching as a Distance and Consistency Problem

### 2.1 Why location matters

Data access cost rises with distance through the memory and storage hierarchy. The source document gives these approximate illustrative latencies:

| Access layer | Approximate latency |
|---|---:|
| CPU L1 cache | 0.5 ns |
| CPU L2 cache | 2.5 ns |
| CPU L3 cache | 10 ns |
| Main memory | 100 ns |
| NVMe SSD | 25,000 ns |
| SATA SSD | 100,000 ns |
| HDD | 5,000,000 ns |
| Same-datacenter network | 500,000 ns |
| Cross-region network | 150,000,000 ns |

These values are approximate, not universal benchmarks. The architectural point is that placing data nearer to the consumer can remove substantial latency.

### 2.2 Replica or projection

A cache may contain:
- A copy of authoritative rows.
- A derived representation, such as HTML, an aggregate, or an API response.

Both are secondary, less-durable representations. The same questions apply: how quickly can they be refreshed, how stale may they become, and what happens when they are unavailable?

### 2.3 Consistency spectrum

| Strategy | Consistency tendency | Latency | Complexity |
|---|---|---|---|
| Synchronous invalidation | Stronger, if coordinated correctly | Higher write-path cost | High |
| Event-driven async invalidation | Eventual | Low on reads | Medium–high |
| TTL expiration | Weaker; stale until expiry | Very low on hits | Low |
| Stale-while-revalidate | Configurable staleness | Very low at refresh boundary | Medium |

The labels are tendencies, not guarantees. For example, synchronous invalidation is only as strong as its ordering and failure-handling semantics.

## 3. Core Cache Patterns

### 3.1 Cache-aside (lazy loading)

1. Application checks the cache.
2. On hit, return the cached value.
3. On miss, read the database, populate the cache, then return the value.

**Advantages:** Simple, explicit control, and the application can fall back to the source when the cache is unavailable.

**Risks:** Miss latency, duplicated concurrent loads, and cache logic spread through application code.

### 3.2 Read-through

The application asks the cache for a value; the cache/provider loads it from the source on a miss.

- **Strength:** Cleaner application code and potential provider-level loading optimizations.
- **Trade-off:** Tighter coupling to the cache provider and potentially less flexible fallback logic.

### 3.3 Write-through

The application writes through the cache, which synchronously writes to the database.

- **Strength:** Can keep cache and database aligned on the write path.
- **Trade-off:** Higher write latency and write amplification; correctness still depends on atomicity and failure semantics.

### 3.4 Write-behind (write-back)

The cache acknowledges writes and flushes them to the database asynchronously.

- **Strength:** Low perceived write latency and high write throughput.
- **Risk:** A cache failure before flush can lose acknowledged writes; recovery and durability are difficult.
- **Use cautiously:** Appropriate only when the workload and durability contract explicitly tolerate the risk or another durable log protects writes.

### 3.5 Write-around

Writes go directly to the database; the cache is populated on a later read miss.

- **Strength:** Avoids filling cache with write-heavy data that is rarely read.
- **Trade-off:** First read after a write usually misses.

**Rule of thumb:** Cache-aside is a practical default for web workloads. Write-through may suit small lookup data requiring coordinated updates. Write-behind needs an explicit durability strategy. Write-around is useful for write-heavy, low-read data.

### Cache-aside flow

```text
Request
   |
   v
Check cache ------ hit ------> Return value
   |
  miss
   v
Read source of truth
   |
   v
Populate cache
   |
   v
Return value
```

## 4. Cache Stampedes and Expiration Waves

### 4.1 Thundering herd / hot-key breakdown

When one popular key expires, many requests may miss at once and all fetch the same value from the database. The resulting overload can exhaust connection pools, increase CPU and latency, trigger timeouts, and cascade into application failures.

```text
Hot key expires
      |
      v
Many concurrent cache misses
      |
      v
Duplicate database queries
      |
      v
Pool exhaustion / high latency
      |
      v
Timeouts and cascading failure
```

A cache intended to absorb load can synchronize requests into a concentrated spike.

### 4.2 TTL synchronization / cache avalanche

If a deployment or cache flush populates many keys with the same fixed TTL, they can all expire together. This is a broad miss wave across many keys, unlike the single-hot-key breakdown.

### 4.3 Mitigation strategies

| Technique | How it helps | Limitations |
|---|---|---|
| **TTL jitter** | Adds random variation to expiry times, spreading load | Does not stop a stampede on one hot key |
| **Per-key distributed lock** | Allows one request to refresh while others wait/recheck | Lock expiry, contention, timeout, and crash handling must be designed |
| **Probabilistic early recomputation (PER)** | Refreshes increasingly likely as expiry approaches; smooths refresh load | Requires background-refresh logic and tuning |
| **Hot-key pre-warming** | Refreshes known hot keys before expiry | Requires predicting hot keys and rate-limiting refresh work |
| **Request coalescing / singleflight** | Deduplicates identical in-flight loads; many callers share one result | Scope matters: process-local coalescing does not coordinate separate application instances |
| **Stale-while-revalidate** | Serves a slightly stale value while one refresh runs | Requires an acceptable staleness window |

A distributed-lock example from the source:

```text
SET product:42:lock 1 NX EX 10  # Acquire a lock with a 10s expiry
# Fetch from database and populate cache
DEL product:42:lock             # Release the lock
```

The implementation needs safe release semantics, lock ownership checks, and a fallback when refresh takes longer than expected; a simplistic unconditional delete can release a lock owned by another requester after expiry/reacquisition.

**Operational rule:** Never assume that a hard-coded shared TTL is harmless at scale. Use jitter and select a hot-key mitigation based on workload and acceptable staleness.

## 5. Eviction Policies and Data Structures

### 5.1 Efficient LRU

Scanning all entries to find the oldest is \(O(n)\) and defeats the purpose of a fast cache at large sizes.

The classic \(O(1)\) LRU design combines:
- A hash map from key to linked-list node.
- A doubly linked list ordered by recency.
- Most-recently-used (MRU) at the head; least-recently-used (LRU) at the tail.

| Operation | Expected complexity |
|---|---:|
| Get and promote node | \(O(1)\) |
| Put/update and promote | \(O(1)\) |
| Evict tail | \(O(1)\) |

This structure has memory overhead for pointers. Production implementations may approximate LRU or record access events asynchronously to reduce contention and overhead.

Examples described in the source:
- **Memcached:** approximate LRU within slab classes.
- **Redis:** sampled approximation for `allkeys-lru`.
- **Caffeine:** asynchronous access-event processing to reduce contention.

### 5.2 LRU cache pollution

LRU assumes recently accessed items are more likely to be accessed again. A sequential one-time batch scan violates that assumption: it inserts many cold items and can evict genuinely hot entries.

### 5.3 LFU and W-TinyLFU

- **LFU (Least Frequently Used):** Evicts items with low access frequency; often better when popularity is stable and scan pollution is a concern.
- **Exact frequency tracking:** More expensive to maintain; a min-heap typically makes updates \(O(\log n)\).
- **W-TinyLFU:** Uses a count-min sketch to estimate frequency, an admission policy to prevent one-hit items from polluting the main cache, and a small recency window for new arrivals. The source describes it as achieving near-optimal hit rates for diverse workloads with amortized \(O(1)\) operations.

### 5.4 Policy comparison

| Policy | Behavior | Typical fit |
|---|---|---|
| LRU | Evict least recently used | General-purpose workloads with temporal locality |
| LFU | Evict least frequently used | Stable popularity distributions |
| W-TinyLFU | Combines frequency, recency, and admission | Mixed workloads and high churn |
| FIFO | Evict oldest inserted item | Simple streaming/queue-like use cases |
| Random | Evict a random key | Simple policies or roughly uniform access |
| TTL-only | Remove expired items | Time-bounded entries such as sessions or codes |

Choose based on measured workload and hit rate, not popularity of the policy name.

## 6. Three Classic Cache Failure Modes

| Failure mode | What happens | Main defenses |
|---|---|---|
| **Cache penetration** | Requests for nonexistent keys miss both cache and database repeatedly | Negative caching, Bloom filters, input validation |
| **Cache breakdown** | One highly popular key expires and a large number of requests hit the database | Request coalescing, lock/single refresh, PER, pre-warming, stale-while-revalidate |
| **Cache avalanche** | Many keys expire or the cache becomes unavailable, causing a broad surge of source reads | TTL jitter, staggered warming, rate limiting, circuit breakers |

### Penetration details

Invalid or nonexistent IDs such as `user:-1`, `user:-2`, and so on can bypass the cache on every request.

- **Negative caching:** Cache “not found” briefly.
- **Bloom filter:** Probabilistically screens membership before expensive lookups; it can have false positives but, in the standard configuration described, no false negatives.
- **Input validation:** Reject obviously invalid requests at the API boundary.

Keep these cases distinct: jitter is primarily for synchronized expiry waves, not arbitrary nonexistent-key traffic.

## 7. Cache Invalidation and Freshness

### 7.1 TTL-based (passive) expiration

Each entry expires after a configured time.

**Advantages:** Simple and decoupled from database writes.

**Costs:** Data can remain stale until expiry, and synchronized expiration can cause load spikes.

Practices in the source:
- Add jitter.
- Use tiered TTLs: shorter for hot data, longer for cold data.
- Consider **soft TTL + hard TTL**: refresh after soft expiry while continuing to serve stale data, but stop serving it beyond the hard limit.

### 7.2 Event-driven (active) invalidation

Database writes emit change/invalidation events, and consumers delete or update cache entries.

Patterns:
- **Application dual-write:** Application writes to the database and publishes to a queue.
- **CDC (Change Data Capture):** A tool such as Debezium reads database WAL/binlog changes and publishes events.
- **Database triggers:** Can notify external systems, but the source generally discourages them due to coupling and performance impact.

**Benefits:** Smaller stale windows.

**Risks:** Missed or duplicated events, ordering problems, consumer lag, and coordination complexity. At-least-once delivery requires idempotent invalidation. A transactional outbox or CDC-based pipeline can reduce the dual-write gap; the precise guarantee depends on implementation.

### 7.3 Stale-while-revalidate

Return the existing value immediately while refreshing it asynchronously. In HTTP caching, `stale-while-revalidate=60` permits a stale response for a limited window while revalidation runs.

- **Benefit:** Avoids a latency spike at expiry and preserves availability.
- **Cost:** Some users receive stale data; the allowed stale window must match business requirements.

### 7.4 Invalidate versus update on write

| Strategy | Write behavior | Strength | Risk |
|---|---|---|---|
| **Write-invalidate** | Delete the cached entry | Simple; next read reloads | Next read incurs a miss |
| **Write-update** | Replace cached value | Avoids miss after write | Concurrent updates can race; payload and write-path complexity increase |

Prefer invalidation for complex objects when partial updates are error-prone. Updating a small counter/value can be suitable when the new value can be computed atomically.

### Invalidation pipeline

```text
Application / database write
            |
            v
Source of truth commits
            |
            v
Outbox or CDC / change event
            |
            v
Event consumer
            |
            v
Invalidate or update shared cache
            |
            v
Notify / invalidate local and edge tiers as needed
```

This is a conceptual flow; a production design must define event durability, ordering, retries, idempotency, and what happens while consumers lag.

## 8. Distributed Cache Topologies

### 8.1 Client-side sharding

The application hashes each key and routes it to one of several cache nodes.

- **Pros:** Simple path, no proxy hop.
- **Cons:** Changing the node count can remap many keys; failover and topology management are application concerns.

### 8.2 Redis Cluster / server-side sharding

The source describes Redis Cluster as partitioning the keyspace into **16,384 hash slots**. A key maps to a slot, slots are assigned to nodes, and clients route requests using a cached slot map.

Important constraint: multi-key operations generally require the keys to share a slot. Hash tags can intentionally group related keys, e.g. `{user:42}:profile` and `{user:42}:orders`. Cross-slot transactions are not supported.

### 8.3 Consistent hashing

Traditional modulo hashing (`hash(key) % N`) relocates many keys when the node count changes.

Consistent hashing maps nodes and keys to a ring; a key is assigned to the next node clockwise. Adding/removing a node usually relocates only a fraction of keys instead of nearly all. **Virtual nodes** improve balance by giving each physical node multiple positions on the ring.

### 8.4 Multi-level caches

| Tier | Placement | Approximate latency in source | Common role |
|---|---|---:|---|
| L1 | In-process (Caffeine, Guava) | ~10 ns | Very hot objects within one application instance |
| L2 | Shared local cache | ~1 ms | Shared cache near a host or process group |
| L3 | Remote distributed cache | ~5 ms | Cross-host cache |
| L4 | CDN / edge | ~20–100 ms | Geographic delivery of static or dynamic content |

The latency values are illustrative and environment-dependent.

**Invalidation problem:** A write must reach every relevant tier. Options include sequential invalidation (L1 → L2 → L3 → L4) or broadcasting invalidation events to tiers. Local caches are especially easy to overlook because each application instance may hold its own copy.

## 9. Geographic Distribution and Edge Caching

### 9.1 Physics sets the floor

Light in fiber travels at roughly two-thirds of its vacuum speed. Long cross-region round trips have a substantial theoretical minimum; real routes add routing, congestion, and protocol overhead. Global users therefore benefit from data and compute placed closer to them.

### 9.2 CDN hierarchy

```text
User
 |
 v
Edge Point of Presence (PoP)
 |
 | miss
 v
Regional mid-tier
 |
 | miss
 v
Origin application / object storage
```

Modern edge platforms can also execute code near users, enabling:
- Edge-side includes and fragment caching.
- Stale-while-revalidate.
- Personalized variants keyed by selected cookies or headers.

Personalized caching requires careful cache-key design to avoid serving one user's content to another.

### 9.3 Edge invalidation strategies

- **TTL:** Easy, but not precise.
- **Purge API:** Invalidates a URL; propagation can be rate-limited or eventually consistent across PoPs.
- **Surrogate keys:** Tag multiple responses with a business key (for example, `product:42`) and purge by tag.
- **Versioned URLs:** Publish `style.v2.css` instead of mutating `style.css`. Immutable, versioned assets largely eliminate the need to purge old versions.

For static assets, content/version-addressed URLs are especially robust because changed content receives a new identity.

### 9.4 Geo-replicated cache models

- **Active-active:** Multiple regions accept writes; conflict resolution or mergeable data types may be needed.
- **Active-passive:** One primary region accepts writes while other regions serve read-only replicas.
- **Global datastore + regional cache:** A globally distributed database is the source of truth, with regional caches accelerating reads.

Do not treat a regional cache as authoritative merely because it is close to users.

## 10. Operations, Observability, and Resilience

### 10.1 Cache warming

A cold cache after restart/deployment can sharply increase source traffic.

- **Pre-warming:** Load predicted hot keys before serving normal traffic.
- **Lazy warming:** Populate on demand and accept an initial miss-rate increase.
- **Shadow traffic:** Duplicate selected requests to populate a new cache without using its responses for user-facing results.

Warming must be rate-limited; an aggressive pre-warm job can itself overload the database.

### 10.2 Metrics to monitor

| Metric | What it reveals |
|---|---|
| Hit rate | Whether the cache is effective for the workload |
| Miss rate | Increased source traffic; spikes can indicate synchronized expiry or cold starts |
| p50/p99 latency | Network, server, or lock contention |
| Eviction rate | Memory pressure or a changed access distribution |
| Database connection-pool saturation | Whether misses are exhausting source capacity |
| Replication lag | Staleness risk when replicas back cache loads |

The source mentions >90% as a possible target for hot data and illustrative latency targets of <1 ms local or <5 ms remote. These are not universal SLOs; define targets based on topology and user-facing requirements.

### 10.3 Protect the source of truth

- **Circuit breaker:** Fail fast when downstream errors exceed a threshold; return a degraded response if the product permits it.
- **Rate limiter:** Cap database load, potentially per key or user.
- **Bulkhead:** Isolate services or workloads so one stampede does not consume all shared capacity.
- **Graceful degradation:** Decide in advance whether stale data, partial results, or an explicit error is acceptable for each endpoint.

A cache outage should not automatically turn into a database outage.

## 11. Decision Framework

### 11.1 When to cache

| Workload | Suggested approach from the source | Key caution |
|---|---|---|
| Read-heavy, rarely changing data | Cache-aside with longer TTL | Still plan for expiry and cold starts |
| Read-heavy, moderately changing catalog | Cache-aside plus event invalidation | Handle event lag and retries |
| Write-heavy, rarely read telemetry/audit data | Often no cache; write-around | Avoid polluting the cache |
| Strongly consistent financial data | Cache cautiously, short TTL/read-through where appropriate | A cache may be inappropriate for authoritative balance reads |
| Global users | CDN plus regional cache | Define invalidation and regional consistency |
| Real-time collaborative data | Not a traditional cache-only problem | May require replicated data structures and conflict handling |

### 11.2 Technology selection

| Requirement | Technology cited in source | Why it may fit |
|---|---|---|
| Simple key-value, multiple languages | Memcached | Minimal overhead and established design |
| Rich data structures / persistence options | Redis | Strings, hashes, sorted sets, streams |
| High availability / failover | Redis Cluster or Sentinel | Replication and promotion mechanisms |
| In-process JVM cache | Caffeine | W-TinyLFU and efficient local caching |
| In-process Go cache | Ristretto | High-performance, memory-bounded caching |
| Geographic edge | Cloudflare, Fastly, Akamai | Global PoP infrastructure |

Selection also depends on workload, operational maturity, durability needs, failure behavior, and total cost—not just raw hit latency.

## 12. Practical Design Checklist

### Correctness and freshness
- [ ] Identify the authoritative source of truth.
- [ ] Define acceptable staleness per data type and endpoint.
- [ ] Choose TTL, active invalidation, write-update, or a combination.
- [ ] Make event consumers idempotent and define retry/ordering behavior.
- [ ] Document what happens when invalidation is delayed or lost.
- [ ] Design invalidation for every cache tier, including per-process L1 caches.

### Stampede and overload protection
- [ ] Add TTL jitter to avoid synchronized expirations.
- [ ] Protect known hot keys with coalescing, early refresh, locks, or pre-warming.
- [ ] Cache negative results for appropriate short periods.
- [ ] Consider Bloom filters or request validation for invalid-key traffic.
- [ ] Rate-limit source reads and define circuit-breaker behavior.
- [ ] Ensure cache warm-up cannot overwhelm the database.

### Performance and capacity
- [ ] Measure hit rate, miss rate, p99 latency, evictions, and source pool saturation.
- [ ] Match eviction policy to real access patterns; test sequential scans.
- [ ] Estimate memory overhead, including metadata and pointers.
- [ ] Test node addition/removal and key redistribution.
- [ ] Define cold-start and cache-node-failure behavior.

### Edge and geography
- [ ] Define cache keys carefully for personalized responses.
- [ ] Prefer versioned immutable URLs for static assets.
- [ ] Define purge propagation expectations for edge content.
- [ ] Decide whether regions are active-active or active-passive.

## 13. Compact Reference: Problem → Defense

| Problem | First defenses to consider |
|---|---|
| Many keys expire together | TTL jitter, staggered warming, rate limits |
| One hot key expires | Singleflight, lock/single refresh, PER, stale-while-revalidate |
| Invalid keys bypass cache | Input validation, negative caching, Bloom filter |
| One-time scans evict hot items | LFU/W-TinyLFU, admission policy |
| Cache node or cluster fails | Failover design, source protection, circuit breaker, graceful degradation |
| Cache tiers disagree | Explicit invalidation fan-out and event observability |
| Global edge content is stale | Purge APIs, surrogate keys, versioned URLs |
| CDC/event consumer lags | Lag metrics, retries, idempotency, bounded stale policy |

## 14. Glossary

- **Bloom filter:** Probabilistic membership structure; standard configurations have no false negatives but can have false positives.
- **Cache avalanche:** Broad simultaneous expiration or cache failure causing a surge in source requests.
- **Cache breakdown:** A hot key expires and causes a concentrated wave of source reads.
- **Cache penetration:** Nonexistent keys repeatedly miss both cache and database.
- **Cache-aside:** Application manages cache lookup and population on misses.
- **CDC:** Change Data Capture; streams database changes to consumers.
- **Consistent hashing:** Ring-based key placement that limits remapping when nodes change.
- **Count-min sketch:** Probabilistic structure used to estimate item frequencies.
- **Jitter:** Random variation added to TTLs to desynchronize expiration.
- **PER:** Probabilistic Early Recomputation; refreshes before expiry with a probability that rises as expiry approaches.
- **PoP:** Point of Presence, a CDN edge location.
- **Singleflight:** Deduplication of identical in-flight requests.
- **Stale-while-revalidate:** Serve stale data while asynchronously refreshing it.
- **W-TinyLFU:** Windowed Tiny LFU policy combining frequency estimation, admission, and recency.
- **WAL:** Write-Ahead Log; durable record of database changes.
- **Write-behind:** Acknowledge cache writes before asynchronously flushing to the database.

## 15. References Listed in the Source

- Redis documentation: Redis Cluster Specification, LRU Approximation, and Key Eviction Policies.
- Caffeine GitHub Wiki: Design of W-TinyLFU (Ben Manes).
- Cloudflare: Cache invalidation.
- Fastly documentation: Surrogate Keys and Purging.
- Martin Kleppmann, *Designing Data-Intensive Applications* (2017).
- Debezium documentation: Change Data Capture Architecture.
- Martin Fowler: *Patterns of Distributed Systems*.
- AWS documentation: ElastiCache for Redis best practices.
- Google SRE Book: caching, overload, and cascading failures.

---

**Core takeaway:** Treat caching as a deliberate placement and consistency decision. Define freshness requirements first, choose a cache pattern and eviction policy to match the workload, distinguish failure modes, and make overload protection part of the design—not an afterthought.
