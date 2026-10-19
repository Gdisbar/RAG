# Distributed Transactions & Data Consistency

**Audience:** Senior engineers, staff engineers, and system architects  
**Focus:** Architectural patterns, operational mechanisms, consistency guarantees, and engineering trade-offs

---

## 1. Core Architectural Principle

Distributed transactions move consistency management from the database layer into application architecture and operations.

In a monolith, a single transaction manager can provide ACID guarantees across orders, inventory, and payments. In microservices, each service owns its database and transaction boundary. Cross-service workflows must explicitly handle partial failures, asynchronous communication, retries, and inconsistent intermediate states.

| Monolithic architecture | Distributed architecture |
|---|---|
| Single ACID transaction boundary | Independent local transactions |
| Database-managed rollback | Business-level compensation |
| Often synchronous coordination | Often asynchronous coordination |
| Database-managed isolation | Explicit consistency contracts |
| Database recovery mechanisms | Retries, deduplication, reconciliation, monitoring |

**Design objective:** Choose the simplest mechanism that satisfies the business invariant. Do not simulate global atomicity without a concrete requirement.

## 2. ACID and the Loss of a Single Transaction Boundary

ACID properties:

- **Atomicity:** Operations in a transaction commit or abort together.
- **Consistency:** Transactions preserve defined data invariants.
- **Isolation:** Concurrent operations are controlled by the selected isolation level.
- **Durability:** Committed changes survive failures.

Core database mechanisms:

| Mechanism | Purpose |
|---|---|
| Write-Ahead Log (WAL) | Records changes durably for crash recovery |
| Undo log | Preserves prior values for rollback |
| Locks | Coordinate conflicting operations |
| MVCC | Uses versioned snapshots to manage concurrent reads and writes |

Isolation is not binary. Read Uncommitted, Read Committed, Repeatable Read, and Serializable provide different guarantees, and implementations vary by database engine.

When services own separate databases, local ACID remains available, but it does not automatically extend across service boundaries.

**Trade-off:** Independent deployment, scaling, technology choice, and fault isolation are gained at the cost of cross-service coordination and consistency complexity.

## 3. Two-Phase Commit (2PC)

2PC attempts to reproduce global atomicity across databases using a coordinator.

```text
                 +----------------------+
                 | Transaction          |
                 | Coordinator          |
                 +----------+-----------+
                            |
                  Phase 1: PREPARE
                    /       |       \\
                   v        v        v
              +--------+ +--------+ +--------+
              | Order  | |Inventory| |Payment |
              |Service | | Service | |Service |
              +---+----+ +----+---+ +---+----+
                  |           |          |
               YES/NO       YES/NO      YES/NO
                  \\           |          /
                   +----------+---------+
                              |
                   Phase 2: COMMIT/ABORT
                              |
                     All participants
```

### Protocol

1. **Prepare / vote:** The coordinator asks each participant to prepare. Participants perform local work, record intent, acquire necessary locks, and vote `YES` or `NO`.
2. **Commit / abort:** The coordinator commits if all participants vote `YES`; otherwise, it aborts. Participants apply the decision and release locks.

### Failure mode and trade-off

- Participants that voted `YES` may hold locks while waiting for the global decision.
- If the coordinator fails, participants may not know whether to commit or abort.
- A participant cannot safely abort solely because a timeout expires; the coordinator may already have decided to commit.
- Locked resources reduce availability and can cause cascading contention.
- Recovery can require log inspection, heuristic decisions, and manual intervention.

2PC prioritizes atomicity over availability when the outcome cannot be determined safely.

**Appropriate context:** XA/JTA enterprise transactions and specialized distributed database internals. Naive 2PC across independently operated microservices is generally a poor default.

Consensus-based distributed databases combine transaction coordination with replicated logs and failover, mitigating single-coordinator blocking. This is not equivalent to adding retries to ordinary 2PC.

## 4. CAP, PACELC, and Consistency Trade-offs

### CAP

During a network partition, a distributed data store cannot guarantee both:

- **Consistency:** Every read returns the most recent write or an error.
- **Availability:** Every request receives a non-error response.

Partition tolerance is assumed for systems designed to operate despite network failures. CAP is not a permanent "pick two" menu; the trade-off applies during partitions.

### PACELC

PACELC extends the model:

- **If Partition (P):** Trade off Availability (A) against Consistency (C).
- **Else (E):** Trade off Latency (L) against Consistency (C).

| System example in the source | PACELC classification | Principal trade-off |
|---|---|---|
| Spanner | PC/EC | Consistency during partitions; consistency over latency in normal operation |
| Cassandra | PA/EL | Availability during partitions; low latency in normal operation |
| MongoDB (default) | PA/EC | Availability during partitions; consistency preference otherwise |
| DynamoDB | PA/EL | Availability and low latency; eventual consistency by default |

These labels simplify systems with configuration-dependent behavior. Real systems may expose tunable consistency, such as Cassandra's `ONE`, `QUORUM`, and `ALL` read/write levels.

**Engineering implication:** Evaluate consistency per operation and failure condition rather than treating CP/AP as permanent system labels.

## 5. Saga Pattern: Distributed Business Transactions

A saga replaces one global transaction with a sequence of independently committed local transactions.

```text
FORWARD EXECUTION
+-------------+      +------------------+      +----------------+
| Create Order| ---> | Reserve Inventory| ---> | Charge Payment |
|   COMMIT    |      |      COMMIT      |      |  COMMIT/FAIL   |
+-------------+      +------------------+      +----------------+
                                                       |
                                                    on failure
                                                       v
COMPENSATION (reverse dependency order)
+-------------+      +------------------+
| Cancel Order| <--- | Release Inventory|
+-------------+      +------------------+
```

### Compensation semantics

A committed transaction generally cannot be rolled back after other services or external systems have observed its effects. Instead, execute a new business operation that compensates for the earlier action.

| Forward operation | Compensation |
|---|---|
| Create order | Cancel order |
| Reserve inventory | Release reservation |
| Charge payment | Refund payment |
| Ship package | Initiate interception or return |

Compensations do not erase history. They create new business state and may themselves fail.

**Key distinction:** Database rollback reverses uncommitted storage changes; compensation is a business-level workflow involving the same external systems as the forward operation.

### Coordination models

#### Choreography (event-driven)

```text
Order Service --OrderCreated--> Inventory Service
Inventory Service --InventoryReserved--> Payment Service
Payment Service --PaymentProcessed--> Shipping Service
```

- **Advantages:** Loose coupling, autonomous services, no central workflow coordinator.
- **Costs:** Implicit dependencies, harder end-to-end tracing, and more difficult ordering and workflow reasoning.

#### Orchestration (command-driven)

```text
              +--------------------+
              | Saga Orchestrator  |
              +---------+----------+
                        |
             +----------+----------+
             |                     |
             v                     v
      Inventory Service      Payment Service
             |                     |
          Reserved               Failed
             |                     |
             +----------+----------+
                        |
               Compensation commands
                        |
                 Release inventory
                 Cancel order
```

- **Advantages:** Explicit workflow, centralized monitoring, easier failure-path reasoning; useful for long-running processes.
- **Costs:** Additional network hops and coordination overhead; orchestrator availability and scaling must be managed.

The source identifies Temporal, Cadence, Camunda, and Netflix Conductor as orchestration tools, and Kafka-based messaging or AWS EventBridge as choreography infrastructure options.

## 6. Consistency Models

| Model | Guarantee | Typical use or trade-off |
|---|---|---|
| **Linearizability** | Operations appear atomic and respect real-time ordering | Strong single-object semantics; higher coordination latency |
| **Serializability** | Transaction outcome is equivalent to some serial execution | Multi-operation transaction correctness |
| **Sequential consistency** | A global sequential order preserves each process's program order | Does not require cross-process real-time ordering |
| **Causal consistency** | Causally related operations are observed in causal order | Collaborative and social applications |
| **Eventual consistency** | Replicas converge if updates stop | Read scalability; staleness can be unbounded |
| **Read-your-own-writes** | A session observes its preceding writes | User-facing session consistency |

Linearizability and serializability are distinct: linearizability concerns operations and real-time ordering; serializability concerns transaction histories.

The design principle is to **choose the weakest consistency model that satisfies the business invariant**. Stronger guarantees usually require more coordination and can increase latency or reduce throughput.

Eventual consistency does not guarantee a short convergence delay. If the business requires a maximum staleness window, that requirement must be explicitly enforced.

## 7. Production-Grade Saga Implementation

### 7.1 Transactional Outbox

**Problem:** A database update and message publication are separate operations. A crash between them can leave committed data without an event, or publish an event for a transaction that rolls back.

**Mechanism:**

1. Update business data and insert an event into an outbox table in the same local database transaction.
2. A relay publishes outbox records to the message bus.
3. Mark or remove successfully processed records.
4. Optionally use Change Data Capture (CDC) to stream outbox changes from the WAL or binlog.

```text
+---------------------- Local DB Transaction ----------------------+
|  1. Update business row                                          |
|  2. Insert event into outbox                                     |
|  3. COMMIT                                                       |
+------------------------------------------------------------------+
                              |
                              v
                    +------------------+
                    | Outbox / CDC     |
                    | Relay            |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Message Broker   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Consumer Service |
                    +------------------+
```

The outbox closes the database/message dual-write gap. It does not by itself eliminate duplicate delivery or guarantee exactly-once business effects; consumers still need idempotency and deduplication.

### 7.2 Idempotency

At-least-once delivery means handlers may receive the same logical request repeatedly. Processing duplicates must not create additional unintended effects.

Common strategies:

- **Idempotency keys:** Persist unique request IDs and deduplicate retries.
- **Database upserts:** Prevent duplicate row creation.
- **State-machine guards:** Reject invalid or already-completed transitions.
- **Conditional updates:** Apply changes only when preconditions hold.

A decrement of inventory is not inherently idempotent. Repeating it can oversell unless guarded by durable deduplication or an equivalent invariant-preserving design.

### 7.3 Compensation failure and recovery

A failed compensation can leave an intermediate business state, such as a canceled order with inventory still reserved.

- **Retries with exponential backoff:** Retry transient failures.
- **Dead Letter Queues (DLQs):** Isolate repeatedly failing work for investigation and remediation.
- **Immutable saga audit logs:** Record forward actions, state transitions, and compensation attempts.
- **Human escalation:** Alerts, runbooks, reconciliation dashboards, and approval paths where required.

Retries should be controlled by operational policy. Unlimited retry is not a substitute for poison-message handling, alerting, or reconciliation. A DLQ requires active monitoring and remediation.

### 7.4 Saga state machine

```text
[START]
   |
   v
[ORDER_CREATED]
   |
   v
[INVENTORY_RESERVED]
   |
   v
[PAYMENT_CHARGED]
   |
   v
[COMPLETED]

Failure paths may transition through:
[INVENTORY_RELEASED] -> [ORDER_CANCELLED]
[PAYMENT_REFUNDED]   -> [ORDER_CANCELLED]
```

The state machine makes workflow progress durable, enforces valid transitions, and supports recovery after restarts and retries.

## 8. Modern Alternatives

| Pattern | Mechanism | Best fit | Main limitation |
|---|---|---|---|
| **TCC (Try-Confirm-Cancel)** | Reserve resources, then confirm or cancel | High-contention inventory, flash sales, ticketing | Requires explicit reservation semantics and additional business logic |
| **Consensus-based distributed transactions** | Replicated logs and consensus combined with transaction coordination | Distributed ACID requirements | Specialized infrastructure and operational expertise |
| **CRDTs** | Data structures designed to converge under concurrent updates without coordination | Counters, sets, maps, presence indicators | Not suitable for arbitrary invariants such as payment ledger correctness |
| **Event sourcing** | Store events as the source of truth; build read projections | Audit-heavy domains and traceable state transitions | Schema evolution, projection lag, snapshotting complexity |

### Architectural distinctions

- **TCC vs. Saga:** TCC introduces explicit resource reservation. Sagas coordinate committed local transactions and subsequent compensation.
- **Distributed SQL vs. Saga:** Distributed SQL retains transaction semantics across supported data boundaries; sagas coordinate workflows across independent services and external systems.
- **CRDT vs. transaction coordination:** CRDTs fit mergeable data types, not general transactional invariants.
- **Event sourcing vs. state storage:** Event sourcing preserves change history and enables projection rebuilding, but increases operational and modeling complexity.

## 9. Architectural Decision Framework

| Scenario | Recommended pattern | Rationale |
|---|---|---|
| Single service and database | ACID transactions | Simple, correct, and avoids unnecessary distributed complexity |
| Multiple databases; long-running workflow | Saga orchestration | Explicit workflow, compensation, and observability |
| Multiple databases; high contention | TCC or Saga with optimistic locking | Controls reservation and contention risks |
| Greenfield system requiring distributed ACID | Distributed SQL | Preserves transactional guarantees with infrastructure trade-offs |
| Read-heavy workload; staleness acceptable | Eventual consistency with CQRS | Improves read scalability while accepting lag |
| Strictly ordered financial records | Consensus-backed transactions or event sourcing | Supports ordering, traceability, and auditability |

## 10. Operational Readiness

Before adopting sagas, verify that the system has:

- [ ] **Idempotency:** Requests and event handlers tolerate retries and duplicate delivery.
- [ ] **Observability:** Distributed traces correlate execution across service boundaries.
- [ ] **Compensation design:** Each forward step has defined and tested recovery behavior.
- [ ] **Message durability:** Delivery guarantees, deduplication, and replay are understood.
- [ ] **DLQ operations:** Alerts, ownership, and remediation runbooks exist.
- [ ] **State visibility:** Clients can see intermediate states such as `PENDING_PAYMENT`.
- [ ] **Auditability:** Saga transitions and compensation attempts are durably recorded.
- [ ] **Reconciliation:** The system can detect and resolve business invariants violated by prolonged or unrecoverable failures.

Failures include not only network outages, but also process pauses, clock skew, disk failures, configuration drift, and human error.

## 11. Common Simplifications to Avoid

| Simplification | Engineering reality |
|---|---|
| "2PC is dead." | Naive cross-microservice 2PC is problematic; consensus-backed transaction protocols remain relevant. |
| "CAP means pick two." | The consistency/availability trade-off applies during partitions. |
| "Eventual consistency means a short delay." | Convergence time may be unbounded without additional guarantees. |
| "Compensation is rollback." | Compensation is a new business operation that may itself fail. |
| "Sagas are simpler." | They shift complexity into application logic, messaging, recovery, and operations. |
| "The network is the only failure source." | Storage, scheduling, clocks, configuration, and human actions also require failure handling. |

## 12. Architect's Takeaways

1. **Preserve local ACID boundaries.** Use database transactions where the work fits within one database.
2. **Make consistency explicit.** Define invariants, permitted intermediate states, and required guarantees per operation.
3. **Use the right coordination model.** Choose sagas for independent long-running workflows; consider distributed SQL when global transactional guarantees are essential.
4. **Treat messaging as a reliability problem.** Combine the outbox pattern with idempotent consumers, durable state transitions, and recovery procedures.
5. **Engineer failure paths.** Coordinator loss, duplicate messages, failed compensation, prolonged partitions, and manual reconciliation are architectural concerns.
6. **Make trade-offs measurable.** Select consistency, availability, latency, throughput, and operational complexity according to business requirements.

**Bottom line:** Determine which consistency guarantees the business needs, then choose the simplest architecture that can deliver them reliably at acceptable operational cost.

## References listed in the source document

1. Brewer (2000), "Towards Robust Distributed Systems" (CAP).
2. Gilbert and Lynch (2002), "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services."
3. Abadi (2010), "Consistency Tradeoffs in Modern Distributed Database System Design: CAP is Only Part of the Story."
4. Garcia-Molina and Salem (1987), "Sagas."
5. Martin Kleppmann (2017), *Designing Data-Intensive Applications*.
6. Sam Newman (2021), *Building Microservices*, 2nd Edition.
7. Martin Fowler (2005), work on Event Sourcing and CQRS.
8. Google (2012), Spanner whitepaper.
9. CockroachDB architecture documentation.
10. Temporal.io documentation.
