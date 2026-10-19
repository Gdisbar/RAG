# Database Architecture & Scaling

> **Audience:** Senior engineers, staff architects, and technical leads  
> **Purpose:** A practical reference for scaling databases from a single optimized node to replicated and sharded architectures, with emphasis on connection handling, indexing, replication, failover, and operational trade-offs.

## 1. Executive Summary

The source's central principle is **optimize a single node before adding distributed complexity**. A well-tuned database can be simpler, faster, and easier to operate than a cluster. Distribution introduces network latency, coordination, more failure modes, and more complicated recovery.

The source's recommended progression is:

1. Optimize queries, connections, indexes, and maintenance.
2. Add replicas and caching when reads are the bottleneck.
3. Consider partitioning for large tables within one database.
4. Shard only when single-node capacity and simpler read-scaling techniques are exhausted.
5. Design high availability around consensus, failover safety, and fencing.

## 2. The Baseline: Optimize Before Distributing

Adding nodes trades operational simplicity and local memory access for capacity, network latency, coordination, and more complex failure handling.

Before scaling out, measure the bottleneck:
- Slow or inefficient queries.
- Too many concurrent database connections.
- Excessive or unused indexes.
- Poor vacuum/autovacuum behavior and table/index bloat.
- CPU, memory, disk I/O, locks, or WAL pressure.
- Read load versus primary write capacity.

**Decision rule:** Do not shard simply because latency rises at a particular query rate. Diagnose the limiting resource and exhaust lower-complexity remedies first.

## 3. Connection Architecture and Pooling

### 3.1 PostgreSQL's process-per-connection model

PostgreSQL creates a dedicated backend process for each client connection. The high-level flow is:

```text
Application
    │ TCP connection
    ▼
PostgreSQL postmaster
    │ authenticate + fork()
    ▼
Backend process ── executes SQL
```

Each backend consumes memory and operating-system scheduling resources. Large numbers of concurrent connections can therefore consume substantial RAM and CPU even before query work becomes the main cost.

### 3.2 PgBouncer and connection pooling

PgBouncer sits between applications and PostgreSQL, keeping a smaller pool of persistent backend connections. Applications borrow a backend for database work, then return it to the pool.

```text
Many application connections
            │
            ▼
        PgBouncer
            │ bounded pool
            ▼
     PostgreSQL backends
```

Pooling reduces backend churn and makes database resource usage more predictable.

### 3.3 Pooling modes and session-state risks

| Mode | Connection lifetime | Appropriate fit | Main risk |
|---|---|---|---|
| Session | Bound for the entire client session | Admin tools, migrations, session-dependent apps | Less pooling benefit |
| Transaction | Returned after each transaction | Typical web/API workloads | Session state may not persist |
| Statement | Returned after each statement | Narrow, simple statement workloads | Multi-statement transaction assumptions break |

Transaction pooling can break features that depend on one persistent backend session, including temporary tables, session-level `SET` state, advisory locks held across transactions, `LISTEN/NOTIFY`, and some prepared-statement patterns.

**Before enabling transaction pooling:** audit application code, ORM behavior, drivers, and queries for session-state dependencies.

### 3.4 Other approaches

- **PgPool-II:** Pooling plus additional capabilities such as load balancing and query caching, with more complexity.
- **Application-side pools:** Reuse connections within an application process, but still create PostgreSQL backend processes.
- **Serverless connection proxies/poolers:** Help prevent cold starts and bursty concurrency from exhausting database connection limits.

The source notes that PostgreSQL also uses background workers for parallel queries, logical replication, autovacuum, and other maintenance; this is separate from the client process-per-connection model.

## 4. Indexing, Heap Storage, and MVCC

### 4.1 Index lookup mechanics

PostgreSQL stores table rows in heap pages and indexes separately, commonly as B-trees. A typical index lookup traverses the index to find a tuple identifier (TID), then fetches the corresponding heap page.

```text
Query predicate
      │
      ▼
 B-tree index lookup
      │ TID (block, offset)
      ▼
 Heap page fetch
      │
      ▼
 MVCC visibility check
      │
      ▼
    Result row
```

The heap fetch and visibility check can be significant costs, especially when lookups touch many random pages.

### 4.2 MVCC and dead tuples

PostgreSQL's Multi-Version Concurrency Control (MVCC) lets readers and writers operate concurrently using row versions.

Conceptually, an update:
1. Marks the previous row version as deleted for the relevant transaction horizon (`xmax`).
2. Writes a new row version with a new creation transaction ID (`xmin`).
3. Updates relevant indexes.
4. Leaves obsolete versions for VACUUM to reclaim when safe.

The database checks row visibility against the transaction's snapshot. HOT (Heap-Only Tuple) chains can help some updates avoid index changes when the required conditions are met.

**Operational consequence:** Autovacuum and vacuum tuning matter for storage reclamation, visibility-map maintenance, and stable query performance.

### 4.3 Covering indexes and index-only scans

A covering index stores additional columns needed by a query:

```sql
CREATE INDEX idx_users_covering
ON users(user_id) INCLUDE (email, status);
```

A query selecting only indexed columns may be able to avoid heap fetches. However, PostgreSQL still needs visibility information. An index-only scan avoids a heap visibility check only when the visibility map indicates that the relevant heap page is all-visible.

**Trade-off:** Covering indexes can improve read performance but consume disk and add work to inserts, updates, deletes, and WAL generation. Their benefit depends on query patterns, table churn, and vacuum effectiveness.

### 4.4 Index write cost, page splits, and bloat

Each extra index must be maintained as rows change. Inserts and updates can generate index writes and WAL; full B-tree pages can split, requiring additional page and parent-node changes. Over-indexing can reduce write throughput and increase storage use.

Maintenance options mentioned in the source include:
- Remove unused indexes using usage statistics.
- Tune autovacuum to match table churn.
- Use `REINDEX` or `REINDEX CONCURRENTLY` where appropriate.
- Consider `pg_repack` for online restructuring.

**Rule of thumb:** Choose indexes from measured query patterns and read/write ratios, not from the assumption that more indexes are always better.

## 5. Replication: Read Scaling and Write Costs

Replication copies changes from a primary to replicas, commonly by shipping and replaying Write-Ahead Log (WAL) records. It is useful for read distribution and availability, but it does not split the primary's write workload.

```text
                    ┌──────────► Read replica 1
Application ──► Primary ─ WAL ──► Read replica 2
                    └──────────► Read replica 3
                      writes happen at primary
```

Every additional replica adds WAL network traffic and replay work. A replica fleet can therefore increase the system's total work without relieving a primary write bottleneck.

### 5.1 Replication methods

| Method | Mechanism | Typical fit |
|---|---|---|
| File-based WAL shipping | Copy completed WAL files | Disaster recovery / offsite standby |
| Streaming replication | Stream WAL as it is generated | Hot standby and read scaling |
| Synchronous streaming | Wait for replica acknowledgement | Stronger durability requirements |
| Logical replication | Replicate selected row-level changes | Selective tables or cross-version scenarios |

### 5.2 Synchronous versus asynchronous

- **Asynchronous:** The primary acknowledges after its local commit point and sends changes to replicas in the background. This reduces write latency but allows lag and possible loss of recently acknowledged transactions if the primary fails before they reach a surviving replica.
- **Synchronous:** The primary waits for the configured replica acknowledgement condition. This improves durability guarantees but adds network latency and can reduce throughput or availability when required replicas are unavailable.

The source lists PostgreSQL `synchronous_commit` modes:

| Setting | Source-described behavior |
|---|---|
| `off` | Does not wait for WAL flush |
| `local` | Waits for local flush only |
| `remote_write` | Waits for the replica to receive WAL into its OS cache |
| `on` | Waits for the replica to flush WAL |
| `remote_apply` | Waits for the replica to apply WAL |

The source also describes quorum commit configurations, such as waiting for any two acknowledgements from a configured group of three synchronous standbys.

### 5.3 Replication lag

Possible causes include slower replica hardware, long-running queries delaying WAL replay, bursts saturating network or disk, and contention from maintenance work.

Useful monitoring surfaces listed in the source:
- `pg_stat_replication` on the primary.
- `pg_stat_wal_receiver` on replicas.
- `pg_last_xact_replay_timestamp()` compared with current time.

Mitigations include adequate replication connections, hot-standby feedback where appropriate, and replication slots to retain WAL until consumers catch up. These choices have their own resource and retention implications.

**Read-your-writes concern:** Applications may need to route a user to the primary or apply a consistency-aware routing policy after a write, rather than immediately reading from a lagging replica.

## 6. High Availability, Consensus, and Fencing

### 6.1 Why ping-based failover is unsafe

A missed heartbeat does not prove that the primary crashed. A network partition may leave the primary alive and accepting application writes while a replica cannot reach it. Promoting the replica based only on failed pings can create two primaries and divergent histories.

### 6.2 Consensus-backed failover

The source describes a common self-managed architecture using Patroni with etcd (or Consul/ZooKeeper) as a distributed configuration/coordination store.

```text
 PostgreSQL primary + Patroni ─┐
                               ├── Consensus store (etcd)
 PostgreSQL replica + Patroni ─┘
          │
          └─ promotion requires valid cluster state / lease
```

A primary renews a lease through the consensus store. If it cannot maintain the lease, it should step down; a replica should not promote without the required quorum-backed approval. This avoids relying on one node's subjective view of network reachability.

### 6.3 Fencing and STONITH

Consensus coordination should be paired with a way to prevent an old primary from continuing to accept writes after losing authority. Fencing methods in the source include:
- STONITH: power off the old node through out-of-band management.
- Cloud/API controls: revoke access or detach resources.
- Load-balancer changes: remove the old primary from the write path.

A zombie primary that remains reachable by cached application connections can still create divergence if it is not effectively fenced.

Managed HA offerings can automate many failover operations, but the source highlights trade-offs in configurability, observability, cost transparency, and vendor lock-in.

## 7. Sharding: When and Why

Sharding divides a dataset across nodes, so each node stores only a subset. It is appropriate when vertical scaling is exhausted, one primary cannot sustain required write throughput, data no longer fits economically on one node, or simpler read scaling is insufficient.

**Do not shard solely to scale reads**; replicas and caches are simpler options in many workloads.

| Dimension | Replication | Sharding |
|---|---|---|
| Data placement | Full or near-full copies | Subsets distributed across shards |
| Write scaling | Does not distribute primary writes | Can distribute writes across shards |
| Read scaling | Distributes reads across copies | Helps for queries routed to a single shard |
| Query complexity | Usually low | Higher for cross-shard queries |
| Operational complexity | Moderate | High |

### 7.1 Sharding strategies

**Hash sharding**

```text
shard = hash(user_id) % num_shards
```

- Pros: Often distributes keys evenly.
- Cons: Range queries may touch many shards; changing shard count can require extensive data movement.

**Range sharding**

```text
Shard 1: user_id 1–1,000,000
Shard 2: user_id 1,000,001–2,000,000
```

- Pros: Efficient range queries and potential for splitting hot ranges.
- Cons: Sequential or skewed access patterns can hotspot one shard.

**List sharding**

```text
Shard 1: country = 'US'
Shard 2: country = 'EU'
```

- Pros: Locality by category.
- Cons: Uneven category sizes can create uneven load.

**Composite sharding** combines strategies, for example distributing users by hash while organizing time-series data by range within a shard.

### 7.2 Cross-shard queries

A router/coordinator may need to parse a query, identify shards, issue subqueries, merge results, and handle sorting, limits, and joins.

- **Point query with shard key:** Usually routes to one shard.
- **Global aggregate:** May query all shards; latency can be bounded by the slowest participant plus coordination.
- **Cross-shard joins:** Can be expensive or unsupported; denormalization or application-level joins may be needed.

### 7.3 Technology examples from the source

| Option | Architecture / source characterization |
|---|---|
| Citus | PostgreSQL extension with coordinator and worker nodes; useful for distributed PostgreSQL workloads |
| Vitess | MySQL-focused proxy/sharding layer |
| Application-level sharding | Maximum routing control, but application owns substantial complexity |
| Spanner / CockroachDB | Distributed SQL systems with built-in partitioning and replication |
| PgShard / MySQL Fabric | Described in the source as legacy or deprecated options |

### 7.4 Operational costs

Sharding affects much more than query routing:
- Schema migrations must be coordinated across shards.
- Backups and restores multiply in count and complexity.
- Per-shard monitoring, hotspot detection, and rebalancing become necessary.
- Rebalancing requires copying data and safely updating routing while requests are in flight.
- Cross-shard foreign keys are generally difficult or unsupported.
- Multi-shard writes need distributed transaction coordination or application-level workflows.

## 8. Distributed Transactions Across Shards

### Two-phase commit (2PC)

1. **Prepare:** The coordinator asks each shard to prepare; participants may hold resources while awaiting the decision.
2. **Commit or abort:** If all participants prepare successfully, the coordinator tells them to commit; otherwise it aborts.

**Failure mode:** If the coordinator fails after participants prepare but before the final decision is delivered, participants may remain in a prepared state and retain locks/resources until recovery or intervention.

### Saga pattern

A saga breaks a cross-shard workflow into local transactions, each with a compensating action.

```text
Local transaction A ──► Local transaction B ──► Local transaction C
        │ failure later?
        ▼
Compensate B ◄──────── Compensate A
```

A saga improves flexibility and avoids a single global atomic commit, but it provides compensation and eventual convergence rather than full ACID atomicity across all steps.

## 9. Modern Nuances and Alternatives

### Read replicas are not caches

| Dimension | Read replica | Cache |
|---|---|---|
| Freshness | Can lag behind primary | Controlled by TTL/invalidation strategy |
| Query model | Full SQL, subject to replica state | Usually key/value or specialized access |
| Durability | Database storage | Often treated as ephemeral/rebuildable |
| Typical fit | Read-heavy database workloads | Hot keys, bursts, sessions when semantics permit |

The source warns against using lagging replicas for session data that requires strong freshness.

### Managed databases

Managed services reduce operational burden for replication, backups, and failover. Trade-offs can include proprietary features, less control over low-level tuning and diagnostics, cost complexity, and service-specific limits.

### Serverless workloads

Cold starts and high function concurrency can create connection storms. Options in the source include managed connection proxies, a PgBouncer service/sidecar, and capping application concurrency to match database capacity.

### Native partitioning versus sharding

PostgreSQL native partitioning divides one logical table into physical partitions within the database. Partition pruning can avoid scanning irrelevant partitions while retaining a simpler operational model than distributing data across independent database nodes.

## 10. Scaling Decision Framework

### Phase 1 — Baseline optimization

- [ ] Audit connection count and pooling mode.
- [ ] Check session-state dependencies before transaction pooling.
- [ ] Use `EXPLAIN (ANALYZE, BUFFERS)` for slow queries.
- [ ] Add indexes based on measured read patterns; remove unused indexes.
- [ ] Tune autovacuum for table churn and monitor bloat.
- [ ] Review `work_mem`, `shared_buffers`, and `effective_cache_size`.
- [ ] Consider prepared statements for repeated query patterns.

### Phase 2 — Read scaling

- [ ] Add asynchronous read replicas if the read workload warrants them.
- [ ] Define read-your-writes routing behavior.
- [ ] Cache hot data at the application layer where semantics permit.
- [ ] Use materialized views for expensive repeated aggregates.
- [ ] Consider native partitioning for large/time-series tables.

### Phase 3 — Write scaling

- [ ] Batch or defer writes where possible.
- [ ] Archive or cold-store old data where appropriate.
- [ ] Evaluate Citus, application-level sharding, or distributed SQL.
- [ ] Explicitly design cross-shard query and transaction behavior.

### Phase 4 — High availability

- [ ] Use consensus-backed failover for self-managed deployments or a managed HA service.
- [ ] Implement and test fencing so an old primary cannot keep accepting writes.
- [ ] Practice failover and recovery regularly.
- [ ] Monitor replication lag, quorum/lease health, connection saturation, vacuum progress, index bloat, and lock contention.

## 11. Architect's Summary

1. **Optimize before distributing.** Diagnose the bottleneck before adding nodes.
2. **Connections have a cost.** Pooling reduces backend churn, but transaction pooling imposes session-state constraints.
3. **Indexes are a read/write trade-off.** They improve selected reads but consume space and add write/WAL work.
4. **Replication primarily helps reads and availability.** It adds replication work and does not remove the primary's write bottleneck.
5. **Failover needs coordination and fencing.** Pings alone cannot safely distinguish crashes from partitions.
6. **Sharding is a major architectural commitment.** It can scale writes, but raises the cost of queries, migrations, backups, integrity, and transactions.
7. **Try simpler options first.** Query tuning, pooling, caching, replicas, materialized views, and native partitioning may solve the problem with less complexity.
8. **Make operations observable.** Track lag, pool saturation, bloat, vacuum, WAL, lock contention, and per-shard hotspots.

## 12. Glossary

- **Backend process:** PostgreSQL OS process handling a client connection.
- **B-tree:** Ordered tree structure used for efficient indexed lookup.
- **Covering index:** Index containing columns needed by a query.
- **Heap:** PostgreSQL's main table-row storage.
- **HOT:** Heap-Only Tuple optimization for eligible updates.
- **MVCC:** Multi-Version Concurrency Control.
- **PgBouncer:** PostgreSQL connection pooler.
- **Quorum:** Minimum votes required for a distributed decision.
- **Replication lag:** Delay before primary changes are replayed on a replica.
- **Sharding:** Distributing subsets of data across nodes.
- **STONITH:** Fencing method that powers off a node.
- **TID:** Tuple identifier, a heap location pointer.
- **VACUUM:** PostgreSQL maintenance for reclaiming dead tuples and updating visibility information.
- **WAL:** Write-Ahead Log used for durability, recovery, and replication.
- **`xmin` / `xmax`:** Transaction identifiers associated with row-version creation and deletion.

## 13. References and Further Reading

The source document lists:
- PostgreSQL documentation on libpq, physical storage, WAL, and high availability/replication.
- Martin Kleppmann, *Designing Data-Intensive Applications*.
- Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm (Raft)*.
- PostgreSQL Wiki, *Number of Database Connections*.
- PgBouncer documentation on pooling modes.
- Patroni documentation on distributed configuration and failover.
- Citus documentation on distributed PostgreSQL architecture.
- PostgreSQL 16 release notes.
