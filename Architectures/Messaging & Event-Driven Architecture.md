

# Messaging & Event-Driven Architecture 

**Focus:** Distributed messaging, Apache Kafka, delivery guarantees, consumer groups, event sourcing, CQRS, and production reliability.

---

## 1. Temporal Coupling: Why Asynchronous Messaging?

### Problem: Synchronous Dependencies

In synchronous communication, services must be available simultaneously.

```text
Client → Service A → Service B → Service C → Service D
```

If any required service fails, the entire request can fail.

**Availability calculation:**

```text
System Availability = A × B × C × D

4 services with 99% availability:

0.99⁴ = 96.06% availability
```

### Solution: Asynchronous Communication

Introduce a durable message broker between services.

```text
              Message Broker
                    │
Producer ──Publish──► Queue/Topic
                         │
                         ▼
                      Consumer
```

The producer publishes a message without waiting for the consumer to complete processing.

**Benefits:**
- Temporal decoupling between services.
- Better fault isolation.
- Independent producer/consumer scaling.
- Durable storage of messages during consumer downtime.

**Trade-off:** Immediate consistency is replaced with eventual consistency.

**Important:** Timeouts, retries, and circuit breakers mitigate synchronous failures but do not eliminate temporal coupling.

---

## 2. Messaging Models: Queue vs Pub/Sub vs Distributed Log

| Feature | Message Queue | Pub/Sub | Kafka Distributed Log |
|---|---|---|---|
| Delivery | One consumer processes a message | Every subscriber receives the event | Depends on consumer groups |
| Message retention | Typically removed after acknowledgment | Depends on broker | Retained according to policy |
| Replay | Generally limited | Depends on system | Supported |
| Primary use | Background jobs | Event broadcasting | Streaming and event processing |
| Example | Email processing | Order notifications | Transaction streams |

### Kafka's Unified Architecture

Kafka is a **distributed, replicated, append-only log**.

```text
                 Kafka Topic
                      │
        ┌─────────────┴─────────────┐
        │                           │
 Consumer Group A             Consumer Group B
        │                           │
   Consumer 1                   Consumer 3
   Consumer 2                   Consumer 4

   Order Processing               Analytics
```

**Key rules:**

- Within one consumer group, a partition is assigned to only one consumer at a time.
- Different consumer groups independently consume the same topic.
- Messages remain available according to retention settings, even after consumption.

**Architectural insight:** Kafka supports queue-like and pub/sub-like behavior through consumer group configuration.

---

## 3. Message Delivery Guarantees

Distributed messaging systems must handle failures, retries, and duplicate messages.

| Guarantee | Behavior | Risk | Typical use |
|---|---|---|---|
| At-most-once | Process zero or one time | Message loss | Noncritical metrics |
| At-least-once | Process one or more times | Duplicates | Business events |
| Exactly-once processing | Avoid duplicate processing effects within supported scope | Complexity | Kafka transactional processing |

### Why Duplicates Occur

```text
Producer → Broker
             │
        Message saved
             │
        ACK lost ✗
             │
Producer retries
             │
        Duplicate message
```

The producer cannot always determine whether the original message was persisted.

### Kafka Exactly-Once Semantics (EOS)

Kafka supports exactly-once processing through:

**1. Idempotent producers**
- Producer ID (PID).
- Per-partition sequence numbers.
- Broker-side duplicate detection.

**2. Kafka transactions**
- Atomic publication across partitions.
- Consumers use `isolation.level=read_committed`.
- Useful for consume → process → produce workflows.

**Limitation:** Kafka EOS does not automatically guarantee exactly-once effects in external databases or APIs.

### Application-Level Idempotency

Use a unique message identifier and a database transaction.

```sql
BEGIN;

INSERT INTO processed_messages(message_id)
VALUES ('msg-999')
ON CONFLICT DO NOTHING;

-- Continue only if a new message ID was inserted.

UPDATE accounts
SET balance = balance + 10
WHERE id = 456;

COMMIT;
```

The implementation must check whether the insert succeeded before executing the update.

**Critical rule:** The idempotency check and business operation must execute within the same database transaction.

Redis can support short-term deduplication, but it should not replace durable database-level idempotency for critical operations.

---

## 4. Transactional Outbox Pattern

### Problem: Dual Writes

Consider:

```text
Service
  ├── Update PostgreSQL ✓
  └── Publish Kafka Event ✗
```

The database update succeeds, but the event is never published.

This creates inconsistent system state.

### Solution: Transactional Outbox

```text
           Application
                │
                ▼
        PostgreSQL Transaction
          ├── Business Update
          └── Outbox Event
                │
                ▼
          CDC / Polling
                │
                ▼
             Kafka
                │
                ▼
             Consumer
```

**Implementation:**

1. Update business data and insert an outbox record in one transaction.
2. A CDC connector or polling worker reads the outbox.
3. Publish the event to Kafka.
4. Mark the event as delivered when using polling.

**Technology:** Debezium can capture database changes from PostgreSQL WAL and publish them to Kafka.

**Benefits:** Reliable event publication, retry support, and no loss of the event record when Kafka is unavailable.

**Important:** Outbox relays may publish duplicates, so consumers still require idempotency.

---

## 5. Kafka Internals: Topics, Partitions, and Ordering

### Kafka Partitioning

Kafka topics contain multiple partitions.

```text
Topic: orders

Partition 0: [E1] [E4] [E7]
Partition 1: [E2] [E5] [E8]
Partition 2: [E3] [E6] [E9]
```

Each partition maintains an ordered sequence of records.

### Partition Assignment

Conceptually, keyed partitioning follows:

```python
partition = hash(message_key) % num_partitions
```

Actual partition selection depends on the producer's partitioner.

**With a message key:**
- Messages with the same key normally reach the same partition.
- Per-key ordering can be maintained.

**Without a key:**
- Messages can be distributed across partitions.
- Global ordering is not guaranteed.

### Partition Reshuffling

Changing partition count changes key-to-partition mapping.

```text
Before:
hash(user_id) % 4 → Partition 2

After:
hash(user_id) % 5 → Partition 4
```

New messages may reach a different partition while older messages remain in the original partition.

**Risk:** Out-of-order processing and inconsistent application state.

**Mitigations:**
- Plan partition capacity early.
- Use custom partitioning strategies when needed.
- Use controlled topic migration for strict ordering.
- Avoid casual partition-count changes on ordered topics.

### Hot Partitions

A frequently occurring key can overload a single partition.

**Solutions:**
- Key salting, if ordering can be sacrificed.
- Dedicated topics for high-volume keys.
- Producer batching.

**Trade-off:** Distributing one key across partitions can break its ordering guarantees.

---

## 6. Kafka Consumer Groups and Rebalancing

### Consumer Group Mechanics

```text
Topic: payments
Partitions: P0, P1, P2, P3

Consumer Group A

Consumer 1 → P0, P1
Consumer 2 → P2, P3
```

**Rules:**

1. A partition belongs to at most one consumer within a group.
2. A consumer may process multiple partitions.
3. Consumers exceeding the partition count remain idle.
4. Different groups independently read all topic partitions.

### Rebalancing

A rebalance occurs when consumer group membership or partition assignment changes.

Examples:
- Consumer joins.
- Consumer crashes.
- Consumer leaves.
- Partition count changes.

**Problem:** Rebalancing can interrupt processing and increase consumer lag.

### Rebalancing Strategies

| Strategy | Behavior |
|---|---|
| Eager | Revokes assignments before redistribution |
| Cooperative | Incrementally transfers partitions |
| Sticky assignment | Minimizes unnecessary partition movement |
| Static membership | Reduces disruption from temporary consumer restarts |

**Important configurations:**

```properties
session.timeout.ms
heartbeat.interval.ms
group.instance.id
```

**Production consideration:** Monitor rebalance frequency, consumer lag, and processing interruptions.

---

## 7. Kafka Producer Acknowledgments and Durability

Kafka producers configure acknowledgment requirements using `acks`.

| Setting | Behavior | Durability |
|---|---|---|
| `acks=0` | No acknowledgment | Lowest |
| `acks=1` | Leader acknowledges | Moderate |
| `acks=all` | Required in-sync replicas acknowledge | Highest |

### Recommended Durability Configuration

```properties
acks=all
min.insync.replicas=2
replication.factor=3
```

This configuration requires at least two in-sync replicas for successful acknowledgment.

**Trade-off:** Stronger durability increases latency and can reduce write availability when insufficient replicas are synchronized.

---

## 8. Event Sourcing and CQRS

### Event Sourcing

Traditional databases primarily store current state.

Event sourcing stores the sequence of changes that produced that state.

```text
UserCreated
    ↓
NameChanged
    ↓
EmailChanged
    ↓
NameChanged
    ↓
Current User State
```

**Advantages:**
- Complete audit history.
- Historical state reconstruction.
- Event replay for debugging.
- New projections from existing events.

**Challenges:**
- Schema evolution.
- Growing event storage.
- Replay overhead.
- Increased system complexity.

**Optimization:** Snapshotting stores periodic state snapshots so rebuilding does not require replaying the entire event history.

### CQRS — Command Query Responsibility Segregation

CQRS separates write and read models.

```text
              Commands
                 │
                 ▼
             Write Model
                 │
                 ▼
              Event Bus
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   Search DB  Analytics  Read API
```

**Write side:** Processes commands and changes state.

**Read side:** Maintains optimized projections for different query requirements.

**Benefits:** Independent read optimization and flexible projections.

**Trade-off:** Read models may lag behind writes, creating eventual consistency.

**When to use:** Audit-heavy systems, complex business domains, and applications requiring multiple specialized read models.

**When to avoid:** Simple CRUD applications without meaningful audit or temporal requirements.

---

## 9. Saga Pattern: Distributed Transactions

A Saga coordinates business transactions spanning multiple microservices.

Instead of one distributed ACID transaction, each service performs a local transaction.

### Saga Approaches

| Choreography | Orchestration |
|---|---|
| Services communicate through events | Central orchestrator coordinates services |
| No central controller | Explicit workflow controller |
| Suitable for simpler flows | Suitable for complex workflows |
| Harder to trace overall execution | Easier to monitor workflow state |

### Compensation Example

```text
Reserve Inventory ✓
        ↓
Charge Payment ✓
        ↓
Create Shipment ✗
        ↓
Refund Payment
        ↓
Release Inventory
```

**Key concept:** A failed distributed workflow triggers compensating transactions rather than a conventional database rollback.

**Challenges:**
- Compensation can fail.
- Operations may be irreversible.
- Retries must be idempotent.
- Distributed tracing is necessary.

---

## 10. Backpressure and Consumer Lag

### Backpressure

Backpressure occurs when producers generate messages faster than consumers process them.

```text
Producer: 10,000 messages/sec
                 │
                 ▼
               Kafka
                 │
                 ▼
Consumer: 6,000 messages/sec

Backlog growth = 4,000 messages/sec
```

**Mitigation techniques:**
- Scale consumers up to available partition parallelism.
- Throttle producers.
- Batch messages.
- Apply producer-side blocking.
- Separate critical and noncritical workloads.

### Consumer Lag

```text
Consumer Lag =
Latest Partition Offset - Committed Consumer Offset
```

Growing lag indicates consumers are not keeping pace.

**Monitor:**
- Lag per partition.
- Lag growth rate.
- Consumer processing latency.
- Rebalance frequency.

The document suggests warning and critical lag thresholds, but production thresholds should depend on throughput and acceptable processing delay.

---

## 11. Dead Letter Queues (DLQ)

### Problem: Poison Pill Messages

A malformed or invalid message repeatedly crashes a consumer.

```text
Consumer reads message
        ↓
Processing fails
        ↓
Consumer restarts
        ↓
Same message retried
        ↓
Infinite failure loop
```

### Solution: Dead Letter Queue

```text
Kafka Topic
     │
     ▼
  Consumer
     │
     ├── Success → Commit Offset
     │
     └── Failure
            │
            ▼
       Retry N times
            │
            ▼
       Dead Letter Topic
            │
            ▼
       Investigation / Replay
```

**Best practices:**
- Limit retries.
- Preserve the original message and error details.
- Monitor DLQ volume.
- Support controlled reprocessing.

For reliable handling, ensure the failed message is durably written to the DLQ before its source offset is committed.

---

## 12. Schema Evolution and Schema Registry

### Problem

A producer changes the message structure, but consumers still expect the previous version.

This can cause deserialization failures.

### Solution: Schema Registry

A Schema Registry centrally manages message schemas and compatibility.

**Supported formats:**
- Avro
- Protobuf
- JSON Schema

### Compatibility Types

| Type | Meaning |
|---|---|
| Backward | New schema reads old data |
| Forward | Old schema reads new data |
| Full | Both backward and forward |
| None | No compatibility enforcement |

**Recommendation from the document:** Use Avro or Protobuf with Schema Registry for structured production event contracts.

---

## 13. Production Kafka Operations

### Cluster Configuration

The document recommends the following baseline:

| Component | Guideline |
|---|---|
| Brokers | 3 or more |
| Replication factor | 3 |
| Minimum in-sync replicas | 2 |
| Storage | Dedicated SSD/NVMe |
| Metadata management | KRaft |
| Partition planning | Account for future growth |

These are starting guidelines, not universal sizing requirements.

### Critical Monitoring Metrics

**Broker:**
- Under-replicated partitions.
- Offline partitions.
- Request latency.

**Producer:**
- Publish errors.
- Retry rates.
- Buffer utilization.

**Consumer:**
- Consumer lag.
- Processing time.
- Rebalance frequency.

**Cluster:**
- Leader elections.
- Controller health.
- Replication health.

### KRaft

KRaft replaces ZooKeeper-based Kafka metadata management.

Benefits include simpler deployment and reduced coordination overhead.

The document recommends KRaft for new Kafka deployments.

---

## 14. Choosing the Right Messaging Technology

| Technology | Best suited for |
|---|---|
| **Apache Kafka** | High-throughput event streaming, replay, event sourcing |
| **RabbitMQ** | Complex routing, traditional queues, request-reply |
| **Apache Pulsar** | Multi-tenancy, geo-replication, tiered storage |
| **NATS JetStream** | Lightweight, low-latency cloud-native messaging |
| **AWS SQS/SNS** | Managed queues and pub/sub |
| **Google Pub/Sub** | Managed, scalable pub/sub |
| **Kafka Streams / Flink** | Stateful stream processing |

### REST vs Asynchronous Messaging

| Requirement | Recommended approach |
|---|---|
| Simple CRUD/read API | REST |
| Immediate response required | Synchronous API |
| Background processing | Queue |
| Multiple downstream subscribers | Pub/Sub |
| High-throughput event processing | Kafka |
| Reliable business events | At-least-once + idempotency |
| Distributed business workflow | Saga |
| Audit and historical reconstruction | Event sourcing |
| Specialized read/write workloads | CQRS |

---

# 15. Key Architectural Takeaways

These are the most important concepts to retain from the entire document.

| # | Concept | Key takeaway |
|---|---|---|
| 1 | **Temporal decoupling** | Durable messaging removes the requirement for producers and consumers to be available simultaneously. |
| 2 | **Kafka architecture** | Kafka is an append-only distributed log supporting queues and pub/sub through consumer groups. |
| 3 | **Delivery guarantees** | At-least-once processing requires idempotent consumers. |
| 4 | **Exactly-once semantics** | Kafka transactions provide EOS within supported boundaries, not automatically across external systems. |
| 5 | **Transactional Outbox** | Persist business updates and event records atomically. |
| 6 | **Partitioning** | Partition keys control distribution and ordering. |
| 7 | **Consumer groups** | Partition count limits consumer parallelism within a group. |
| 8 | **Rebalancing** | Membership changes can disrupt processing and increase lag. |
| 9 | **Event sourcing** | Store immutable events and reconstruct state through replay. |
| 10 | **CQRS** | Separate write and read models when justified by business requirements. |
| 11 | **Saga** | Coordinate distributed transactions using local operations and compensations. |
| 12 | **Backpressure** | Prevent producer throughput from overwhelming consumers. |
| 13 | **DLQ** | Isolate repeatedly failing messages without blocking the entire processing pipeline. |
| 14 | **Schema Registry** | Enforce message contract compatibility across evolving services. |
| 15 | **Observability** | Consumer lag, replication health, and rebalancing are essential production signals. |

---

## Final Mental Model

A production-ready event-driven architecture combines:

```text
                   API / Producer
                         │
                         ▼
                Business Transaction
                         │
                         ▼
                 Transactional Outbox
                         │
                         ▼
                    CDC / Relay
                         │
                         ▼
                Kafka Topic / Log
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
      Consumer Group A          Consumer Group B
            │                         │
            ▼                         ▼
     Idempotency Check          Event Processing
            │                         │
            ▼                         ▼
      Business Logic             Read Projection
            │                         │
       ┌────┴────┐                    ▼
       │         │                Query API
       ▼         ▼
    Success    Failure
       │         │
       ▼         ▼
 Commit Offset  Retry
                 │
                 ▼
                DLQ
```

**The central architectural principle:** A message broker provides durable communication, but production reliability comes from combining it with idempotency, transactional consistency, correct partitioning, failure recovery, schema management, and observability.

This condensed version retains all 15 major technical areas of your original document, with the emphasis on concepts, mechanisms, and engineering trade-offs rather than extended explanations. 
