# Distributed Consensus, Coordination, and Quorum Systems 

> **Audience:** Senior engineers, staff architects, and technical leads  
> **Purpose:** A practical reference to distributed agreement, split-brain prevention, Raft and Paxos, tunable quorums, coordination services, scaling, and operational trade-offs.

## 1. Executive Summary

Distributed systems cannot reliably tell whether a silent peer has crashed, is slow, or is unreachable because of a network partition. Letting nodes make critical decisions independently can create **split-brain**: multiple leaders write contradictory histories.

The core design principles are:

- **Require quorum agreement** for critical decisions.
- **Use majority intersection** to prevent two independent majorities from committing conflicting histories.
- **Understand the latency cost** of strong consistency: consensus typically requires durable local writes and network acknowledgements.
- **Choose consistency deliberately.** Tunable read/write quorums trade latency and availability against the likelihood of stale reads or conflicting versions.
- **Use Raft as a practical default** for crash-fault consensus when its assumptions fit.
- **Design for operations:** disk latency, snapshots, reconfiguration, maintenance, and multi-region RTT matter as much as protocol choice.

## 2. Split-Brain and Failure Ambiguity

When node A stops hearing from node B, it cannot tell locally whether B has crashed, paused (for example, during garbage collection), is overloaded, or is separated by a network failure. These cases are observationally equivalent from A's perspective.

**Split-brain** occurs when isolated groups independently believe they can lead or accept authoritative writes. Reconciling later may not be straightforward: conflicting business actions can require domain-specific rules or human judgment.

```text
             Network partition
          ┌─────────────────────┐
          │                     │
      ┌───▼───┐             ┌───▼───┐
      │ A, B  │      X      │ C, D, E│
      └───┬───┘             └───┬────┘
          │                     │
     2 votes                  3 votes
    no majority             majority
     no leader              can elect
```

**Architectural rule:** Critical actions should require a quorum, not merely a local timeout or a single node's belief that its peer is dead.

## 3. Quorum Mathematics

For a cluster of \(N\) voters, a strict majority is:

\[
Q = \lfloor N/2 \rfloor + 1
\]

Any two strict majorities must intersect. If two majorities were disjoint, their combined size would exceed the total number of nodes. This overlap is the mathematical basis for preventing conflicting majority decisions.

### Majority sizes and failure tolerance

| Nodes | Majority | Maximum failures tolerated while retaining a majority |
|---:|---:|---:|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 2 | 1 |
| 4 | 3 | 1 |
| 5 | 3 | 2 |
| 7 | 4 | 3 |

A 2-node cluster cannot tolerate a partition and still make progress. Three nodes are the practical minimum for tolerating one failure. Odd-sized clusters are often preferred because adding one node to an even-sized cluster may not increase failure tolerance.

```text
N = 5, quorum = 3

Partition A: [A, B]       Partition B: [C, D, E]
      2 votes                    3 votes
    cannot elect                 can elect
```

## 4. FLP Impossibility

The FLP result says that in a fully asynchronous system with even one faulty process, no deterministic consensus algorithm can guarantee both:

- **Safety:** correct nodes never decide on conflicting values.
- **Liveness:** the protocol always eventually terminates.

Practical algorithms work under additional assumptions or probabilistic mechanisms, such as randomized election timeouts and partial synchrony. In practice, protocols preserve safety while progress can stall under sufficiently adverse timing or connectivity conditions.

**Trade-off:** Consensus is not magic certainty under every possible network schedule; it is a carefully specified safety guarantee with operational assumptions for making progress.

## 5. Raft: Leader Election and Replicated Logs

Raft separates consensus into **leader election**, **log replication**, and **safety**.

### 5.1 Leader election

Each follower uses a randomized election timeout. A follower whose timeout expires first becomes a candidate and requests votes. Randomization reduces synchronized elections and repeated ties.

A **term** is a monotonically increasing epoch that helps nodes identify stale leaders and candidates. A **pre-vote** extension asks peers whether an election would likely succeed before incrementing the term, reducing unnecessary disruption from transient network blips.

```text
Client
  │ command
  ▼
Leader ───── AppendEntries ─────► Followers
  │                                  │
  │ ◄────── acknowledgements ─────────┘
  │
  ├─ majority (including leader) acknowledges
  ▼
Entry committed → apply to state machine → reply to client
```

### 5.2 Write path

1. Client sends a command to the leader.
2. Leader appends the command to its local log as an uncommitted entry.
3. Leader sends `AppendEntries` RPCs to followers.
4. Followers append the entry and acknowledge.
5. After a majority has acknowledged, the leader marks the entry committed according to Raft's rules.
6. The leader applies the committed entry to its state machine and informs followers of commit progress.
7. The leader responds successfully to the client.

**Trade-off:** A successful write waits for quorum acknowledgement, adding network and durable-storage latency in exchange for strong ordering and fault tolerance.

### 5.3 Log matching and follower catch-up

`AppendEntries` includes previous-log index/term information. A follower rejects entries if its log does not match the expected prefix. The leader then adjusts `nextIndex` and retries until it finds a verified common prefix.

Production implementations optimize catch-up by using rejection hints or indexes to skip back efficiently. The invariant is more important than the specific retry strategy: **establish a common log prefix before appending new history.**

### 5.4 Heartbeats and safety

Heartbeats are empty `AppendEntries` requests in the basic Raft model. They help followers learn that a leader is active and reset election timers.

Key safety properties:

- **Election Safety:** At most one leader is elected in a term.
- **Leader Append-Only:** A leader does not overwrite or delete its own log entries.
- **Log Matching:** Matching index and term imply matching preceding log history.
- **Leader Completeness:** Committed entries appear in leaders of later terms.
- **State Machine Safety:** Nodes do not apply different commands at the same log index.

These guarantees apply within a Raft group. Systems using multiple groups still need coordination for transactions that span groups.

## 6. Paxos Compared with Raft

Paxos uses proposal numbers and an acceptor quorum. Its core flow is often described as prepare/promise followed by accept. Competing proposers and the distinctions among proposer, acceptor, learner, and leader make the protocol difficult to implement and reason about.

| Dimension | Paxos | Raft |
|---|---|---|
| Core design | Proposal-number and quorum based | Explicit leader, terms, replicated log |
| Understandability | Often considered harder to teach and implement | Designed for understandability |
| Performance | Comparable in many leader-based deployments | Comparable in many leader-based deployments |
| Practical selection | Existing systems, specific verification or design needs | Common default when crash-fault consensus fits |

**Decision rule:** Prefer the protocol your team can operate and validate correctly. Raft is a reasonable default for new crash-fault replicated-state systems; existing constraints may justify Paxos-family protocols.

## 7. Tunable Read/Write Quorums

For a replicated item:

- \(N\): number of replicas.
- \(W\): replicas that must acknowledge a write.
- \(R\): replicas that must respond to a read.

The overlap condition is:

\[
W + R > N
\]

When this holds, every read quorum intersects every write quorum. This is an important overlap property, but it does **not by itself guarantee linearizability**: version selection, concurrent writes, replica behavior, and protocol semantics also matter.

| Pattern | Example \(N,W,R\) | Typical trade-off |
|---|---:|---|
| Quorum read/write | 3, 2, 2 | Read/write quorum overlap; more coordination |
| Fast writes | 3, 1, 3 | Low write acknowledgement threshold; broad reads |
| Fast reads | 3, 3, 1 | Broad write acknowledgement; low read latency |
| Eventual-style | 3, 1, 1 | Low latency; no guaranteed overlap |
| Sloppy quorum | Depends on implementation | More availability using temporary/non-preferred nodes; reconciliation required |

### Sloppy quorum and hinted handoff

When preferred replicas are unavailable, a system may accept a write on a temporary node and record a hint. When the intended replica returns, the temporary node forwards the data. This improves availability during a partition but can create a period of inconsistency.

### Conflict resolution

- **Last-write-wins (LWW):** Selects a version using timestamps; clock skew can produce surprising results.
- **Vector clocks:** Help identify causal relationships and concurrent versions; they detect conflicts but do not automatically resolve them.
- **Application-level merge:** Uses domain rules, such as set union for some shopping-cart representations.

```text
Write on replica A ──┐
                     ├─ concurrent versions ─► read/reconcile
Write on replica B ──┘                         ├─ LWW
                                               ├─ vector-clock detection
                                               └─ application merge
```

## 8. Coordination Services in Practice

### etcd

- Raft-based distributed key-value store used for Kubernetes cluster state.
- Provides watches for change notifications and leases for time-bound registrations.
- Operational attention: keep values appropriately small, and manage compaction and defragmentation.

### ZooKeeper

- Uses ZAB (ZooKeeper Atomic Broadcast), a protocol in the Paxos family.
- Common coordination uses include leader election, distributed locks, configuration, and service discovery.
- Runtime pauses and workload characteristics can affect availability and throughput.

### Consul

- Raft-based coordination with service discovery, health checks, and key-value storage.
- Provides DNS-based discovery and supports multi-datacenter deployments with WAN gossip.

| Feature | etcd | ZooKeeper | Consul |
|---|---|---|---|
| Consensus/broadcast | Raft | ZAB | Raft |
| Typical emphasis | Kubernetes state/configuration | Coordination primitives | Service discovery and service networking |
| Watches/notifications | Yes | Yes | Yes |
| Discovery interface | API | Client API | HTTP/DNS |
| Multi-datacenter approach | Typically a single cluster | Deployment-dependent | WAN gossip between datacenters |

Choose based on the primary coordination problem, client ecosystem, consistency semantics, operational expertise, and deployment topology.

## 9. Byzantine Fault Tolerance

Raft and Paxos primarily address crash-stop or crash-recovery faults, not arbitrary malicious or inconsistent behavior.

**Byzantine faults** include a node lying about a write, sending conflicting messages to peers, or corrupting data. PBFT-style systems use multiple rounds of voting and cryptographic techniques. The classic bound is \(3f+1\) nodes to tolerate \(f\) Byzantine faults under the protocol's assumptions.

BFT can incur substantially higher communication overhead and complexity. It is most relevant where nodes cross adversarial trust boundaries or malicious behavior is part of the threat model; it is usually unnecessary for trusted internal infrastructure with ordinary crash-fault assumptions.

## 10. Multi-Raft and Horizontal Scaling

A single Raft group has one leader, which can become a write-throughput and geographic-latency bottleneck. **Multi-Raft** partitions data into ranges or regions, each managed by an independent Raft group with its own leader.

```text
                 Database / keyspace
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Region 1        Region 2        Region 3
      Raft group      Raft group      Raft group
      Leader A        Leader B        Leader C
```

Benefits:
- Spreads leadership and write load across nodes.
- Allows a stalled group to be isolated from unrelated groups.
- Enables horizontal scaling by adding nodes and redistributing ranges.

Challenges:
- Cross-region transactions require extra coordination, often two-phase commit.
- Leader balancing is needed to avoid hotspots.
- Range splitting, merging, and movement require careful orchestration.

## 11. Operational Realities

### Maintenance and quorum headroom

A three-voter cluster tolerates one failure. While one node is down for maintenance, another failure removes the majority and stops quorum-dependent progress. Plan upgrades and maintenance with explicit failure headroom; do not assume the cluster remains resilient to another fault during a rolling change.

### Disk and network latency

Consensus writes can be limited by both durable local storage (such as WAL `fsync`) and the network round trip to a quorum. Cross-region RTT often dominates latency.

```text
Client → leader local durable log → quorum replication/acks
       → commit → state-machine apply → response
```

Mitigations include batching, low-latency storage, appropriate replica placement, and avoiding cross-region consensus unless the durability/consistency requirement justifies the latency.

### Snapshots and compaction

Logs grow continuously. Periodic snapshots capture state-machine state so old log entries can be discarded. Lagging followers can receive a snapshot plus newer log entries rather than replaying the entire history.

**Trade-off:** Snapshotting consumes CPU, disk I/O, and network bandwidth; schedule and size it with operational load in mind.

### Membership changes

Changing cluster membership must preserve quorum safety. Raft joint consensus uses an overlapping configuration transition: first commit a joint old-plus-new configuration, then commit the new configuration. Follow the implementation's supported membership-change procedure rather than changing voter counts informally.

## 12. Consensus, Consistency, and Coordination

| Term | Meaning | Example |
|---|---|---|
| **Consensus** | Agreement on a value among distributed nodes | Leader election |
| **Consistency** | Rules governing what reads may observe after writes | Linearizable or eventual reads |
| **Coordination** | Managing distributed state and actions | Locks, leases, barriers |
| **Consensus algorithm** | Protocol used to reach agreement | Raft, Paxos |
| **Consistency model** | Client-visible read/write guarantee | Linearizable, causal, eventual |

A system can use consensus for one part of its architecture without making every data path strongly consistent. Conversely, a single-node database can provide strong local consistency without running a distributed consensus protocol.

## 13. Decision Framework

| Requirement | Direction suggested by the source |
|---|---|
| Leader election / authoritative configuration | Strong consensus |
| Financial or inventory invariants | Strong consistency and appropriate transaction coordination |
| High-throughput telemetry | Asynchronous replication or a log/streaming system may fit better |
| Global user sessions | Sticky sessions or a shared cache may be simpler |
| Service discovery | Match consistency needs; eventual DNS may suffice in some designs |
| Adversarial multi-party environment | Evaluate Byzantine fault tolerance |
| Large, horizontally scaled keyspace | Consider sharding and Multi-Raft, accounting for cross-shard transactions |

### Example quorum choices

| Scenario | Example \(N,W,R\) | Intent |
|---|---:|---|
| Banking ledger | 5, 3, 3 | Stronger overlap and durability, with higher latency |
| Shopping cart | 3, 1, 1 | Fast updates; requires a reconciliation strategy |
| Product catalog | 3, 1, 2 | Fast write acknowledgements and broader reads |
| Real-time analytics | 3, 1, 1 | Low-latency ingestion with asynchronous aggregation |
| Multi-region database | 5, 3, 3 | Cross-region coordination; potentially high write latency |

These are architectural examples, not universal prescriptions. Actual guarantees depend on the database's implementation, read-repair/version rules, replica placement, failure model, and transaction semantics.

## 14. Architect's Checklist

- [ ] Define which failures are in scope: crashes, partitions, or malicious behavior.
- [ ] State the required consistency and availability behavior during partitions.
- [ ] Choose voter count and quorum size based on tolerated failures.
- [ ] Ensure writes are acknowledged only at the intended commit point.
- [ ] Monitor leader changes, election churn, quorum loss, replication lag, and disk latency.
- [ ] Test partitions, slow nodes, process pauses, and node restarts.
- [ ] Plan snapshots, log compaction, backups, and recovery drills.
- [ ] Use supported membership-change procedures and preserve quorum during maintenance.
- [ ] For Multi-Raft, monitor leader distribution, hotspots, and cross-range transaction latency.
- [ ] Document conflict-resolution semantics for eventually consistent data.
- [ ] Verify whether BFT is actually required by the trust model.

## 15. Glossary

- **AppendEntries:** Raft RPC used for log replication and heartbeats.
- **Byzantine fault:** Arbitrary or malicious node behavior.
- **Commit:** A log entry has reached the protocol's commit condition and is safe to apply.
- **FLP result:** Impossibility result for deterministic consensus in asynchronous systems with a faulty process.
- **Joint consensus:** Raft membership transition requiring quorums from old and new configurations.
- **Majority quorum:** More than half of the voters.
- **Multi-Raft:** Multiple independent Raft groups used to shard a larger system.
- **Pre-vote:** Election extension intended to reduce term inflation from transient failures.
- **Sloppy quorum:** Availability-oriented writes using temporary/non-preferred nodes.
- **Split-brain:** Multiple nodes or groups act as authoritative leaders.
- **Term:** Raft's monotonically increasing election epoch.
- **W + R > N:** Condition for read and write quorums to intersect.
- **ZAB:** ZooKeeper Atomic Broadcast.

## 16. References and Further Reading

The source document lists the following references:

- Ongaro & Ousterhout — *In Search of an Understandable Consensus Algorithm (Raft)*.
- Lamport — *Paxos Made Simple*.
- Fischer, Lynch & Paterson — *Impossibility of Distributed Consensus with One Faulty Process*.
- Castro & Liskov — *Practical Byzantine Fault Tolerance*.
- Martin Kleppmann — *Designing Data-Intensive Applications*.
- etcd, ZooKeeper, and Consul documentation.
- DeCandia et al. — *Dynamo: Amazon's Highly Available Key-Value Store*.
- CockroachDB material on Multi-Raft.
