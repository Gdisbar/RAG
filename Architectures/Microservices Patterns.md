# Microservices Patterns

**Focus:** Microservices resilience, cascading failure prevention, circuit breakers, bulkheads, API gateways, Backend for Frontend (BFF), service mesh, security, and production observability.

This summary preserves the key architectural concepts, implementation mechanisms, configuration examples, trade-offs, and production considerations from your document while removing repetitive explanations.

---

## 1. Cascading Failures: Why Microservices Need Resilience

### Problem: Shared Dependencies and Resources

In distributed systems, a failure in one service can propagate across multiple services.

```text
                  API Gateway
                       |
                       v
                  Order Service
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Payment       Inventory    Analytics
      Service       Service      Service
          |            |            |
          v            v            v
        DB           DB         Slow DB
                                    |
                                    v
                           Thread Pool Exhaustion
                                    |
                                    v
                           Order Service Failure
```

**Why cascading failures happen:**
- Shared thread pools and connection pools.
- Slow downstream dependencies.
- Excessive retries.
- Resource exhaustion.
- Long synchronous service chains.

**Architectural principle:** Failures cannot always be prevented, but their *blast radius* can be contained.

### Control Plane Dependency Trap

The document uses the 2021 Facebook outage to illustrate a critical infrastructure problem: recovery tools depended on the same infrastructure that had failed.

**Key lesson:** Infrastructure recovery mechanisms should not depend entirely on the systems they are intended to recover.

---

## 2. Synchronous vs. Asynchronous Communication

Not every operation can be asynchronous.

**Synchronous operations:**
- Payment confirmation.
- Authentication and authorization.
- Real-time fraud checks.
- Interactive search.

**Asynchronous operations:**
- Email notifications.
- Analytics events.
- Search indexing.
- Background processing.

### Chained Availability

For a synchronous chain:

```text
Client → A → B → C → D → E
```

The document models aggregate availability as:

```text
System Availability = A₁ × A₂ × ... × Aₙ
```

| Individual Service Availability | Services | Combined Availability |
|---|---:|---:|
| 99% | 5 | 95.1% |
| 99.9% | 5 | 99.5% |
| 99.9% | 10 | 99.0% |
| 99.99% | 20 | 99.8% |

**Mitigation:** Reduce unnecessary synchronous dependencies and parallelize independent calls to improve latency.

**Important distinction:** Parallelization reduces waiting time, but if every dependency must succeed, it does not eliminate their combined availability risk.

---

# 3. Circuit Breaker Pattern

A circuit breaker prevents repeated calls to an unhealthy dependency.

### Problem: Timeouts Alone Are Insufficient

Suppose:

```text
Incoming traffic = 1,000 requests/sec
Thread pool      = 100 threads
Request timeout  = 500 ms
```

Using the document's example:

```text
Required concurrency = Arrival Rate × Request Duration

                     = 1,000 × 0.5
                     = 500 concurrent requests
```

Only 100 threads are available.

**Result:** Requests accumulate or fail despite the timeout.

### Circuit Breaker State Machine

```text
                  Failure Threshold
                         |
                         v
        +--------+     +--------+
        | CLOSED |---->|  OPEN  |
        +--------+     +--------+
             ^              |
             |              |
             |       Recovery Timeout
             |              |
             |              v
             |        +-----------+
             +--------| HALF-OPEN |
              Success +-----------+
```

| State | Behavior |
|---|---|
| **CLOSED** | Requests flow normally; failures are monitored |
| **OPEN** | Requests fail immediately or use a fallback |
| **HALF-OPEN** | Limited requests test dependency recovery |

### Failure Detection

The document recommends **sliding-window-based detection** rather than simple cumulative or consecutive failure counters.

Two approaches:

- **Count-based:** Evaluate the last N requests.
- **Time-based:** Evaluate requests within the last N seconds.

### Example Configuration

```yaml
circuit_breaker:
  failure_rate_threshold: 50
  sliding_window_size: 100
  minimum_calls: 20
  open_duration_seconds: 30
  half_open_permitted_calls: 5
  success_threshold: 80
```

These are illustrative settings from the document, not universal production defaults.

### Critical Insight: Monitor Latency Too

A dependency can become extremely slow without producing errors.

```text
Timeout:          2,000 ms
Dependency p99:   1,900 ms
```

An error-only circuit breaker may remain closed.

**Recommended signals:**
- HTTP 5xx errors.
- Connection failures.
- Timeouts.
- p95/p99 latency degradation.

### Fallback Strategies

| Fallback | Behavior | Example |
|---|---|---|
| Fail fast | Return error immediately | Payment unavailable |
| Cached response | Return last known data | Product recommendations |
| Default value | Return safe predefined value | Feature configuration |
| Degraded mode | Disable optional functionality | Hide recommendations |
| Queue for later | Persist work for retry | Analytics processing |

**Key takeaway:** Circuit breakers prevent unhealthy dependencies from continuously consuming application resources.

---

# 4. Bulkhead Pattern

Bulkheads isolate resources so that one failing dependency cannot exhaust the capacity required by other dependencies.

### Without Bulkheads

```text
                 Application
                      |
                Shared Pool
                 100 Threads
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Service A   Service B   Service C
          |
          v
       Very Slow
          |
          v
   Consumes All Threads
          |
          v
    Entire App Suffers
```

### With Bulkheads

```text
                  Application
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Pool A        Pool B        Pool C
     20 Threads    30 Threads    50 Threads
          |            |            |
          v            v            v
      Service A     Service B     Service C
        Slow          Healthy       Healthy
```

If Service A becomes slow, its resource consumption remains limited.

### Implementation Strategies

| Strategy | Mechanism | Trade-off |
|---|---|---|
| Thread pool isolation | Dedicated threads per dependency | Stronger isolation, higher overhead |
| Semaphore isolation | Limit concurrent calls | Lightweight, caller threads still execute |
| Connection pool isolation | Separate database/HTTP pools | Prevents connection starvation |
| Container resource limits | CPU and memory quotas | Service-level isolation |

### Bulkhead + Circuit Breaker

These patterns solve different problems.

| Bulkhead | Circuit Breaker |
|---|---|
| Limits resource consumption | Stops calls to unhealthy dependencies |
| Controls concurrency | Controls failure propagation |
| Protects other dependencies | Protects callers from repeated failures |

### Combined Flow

```text
Incoming Request
       |
       v
  Bulkhead Check
       |
       +---- Full ------> Reject
       |
       v
 Circuit Breaker
       |
       +---- OPEN ------> Fallback / Fail Fast
       |
       v
 Downstream Service
       |
       +---- Success
       |
       +---- Failure ---> Record Failure
```

**Architectural rule:** Use bulkheads to contain resource exhaustion and circuit breakers to stop repeated calls to degraded dependencies.

---

# 5. API Gateway Pattern

An API Gateway centralizes cross-cutting concerns for external client traffic.

### Architecture

```text
                External Clients
                       |
                       v
                  API Gateway
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
   Auth Service    Order Service   User Service
```

### Gateway Responsibilities

| Responsibility | API Gateway | Microservice |
|---|---|---|
| TLS termination | Yes | Usually not at public edge |
| Authentication/JWT validation | Yes | Domain authorization still applies |
| Rate limiting | Yes | Additional internal limits if needed |
| Request routing | Yes | No |
| Load balancing | Yes | No |
| CORS | Yes | Usually centralized |
| Business logic | No | Yes |
| Database operations | No | Yes |

### Gateway Anti-Patterns

**1. God Gateway**

A single gateway accumulates excessive routing, configuration, and coordination responsibilities.

**2. Dumb Pipe**

The gateway only forwards requests without enforcing necessary perimeter protections.

**3. Smart Proxy**

Business logic is implemented inside the gateway, creating coupling between infrastructure and application domains.

### Common Implementations

| Technology | Typical use |
|---|---|
| Nginx | Reverse proxy and routing |
| Kong | API management and plugins |
| Envoy | Layer 7 proxy |
| AWS API Gateway | Managed AWS APIs |
| Azure API Management | Enterprise API policies |
| Istio Ingress | Kubernetes-integrated ingress |

**Key takeaway:** API gateways should handle infrastructure concerns, not domain-specific business logic.

---

# 6. Backend for Frontend (BFF) Pattern

A BFF provides client-specific backend interfaces.

### Problem

Different clients have different requirements.

| Client | Main requirement |
|---|---|
| Mobile | Small payloads and fewer network calls |
| Desktop | Rich data and flexible queries |
| IoT | Minimal payload and lightweight protocols |
| Third-party API | Stable contracts and versioning |

### Architecture

```text
                    Clients
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
      Mobile         Desktop         IoT
        |              |              |
        v              v              v
    Mobile BFF     Web BFF         IoT BFF
        |              |              |
        +--------------+--------------+
                       |
                       v
              Internal Services
              /       |       \
           Orders   Users   Inventory
```

### BFF Responsibilities

- Aggregate multiple backend responses.
- Transform data for client-specific requirements.
- Reduce unnecessary network round trips.
- Implement client-specific caching.
- Apply client-specific rate limits.

### API Gateway vs. BFF

| Feature | API Gateway | BFF |
|---|---|---|
| Scope | Shared API perimeter | Specific client |
| Primary concern | Security, routing, policies | Client-specific data aggregation |
| Ownership | Platform team | Frontend/product team |
| Business data transformation | Minimal | Significant |
| Deployment | Shared infrastructure | Independently managed |

### GraphQL as BFF

GraphQL can allow clients to request only the fields they need.

**Benefits:** Reduced over-fetching and flexible aggregation.

**Challenges:** Schema management, resolver performance, and N+1 queries.

**Key takeaway:** Use BFFs when different client applications have substantially different data-access requirements.

---

# 7. Service Mesh Pattern

A service mesh manages communication between internal microservices.

### North-South vs. East-West Traffic

```text
                  Internet
                     |
                     | North-South
                     v
                 API Gateway
                     |
                     v
                Service A
                     |
                     | East-West
                     v
                Service B
                     |
                     v
                Service C
```

- **North-South:** External client-to-application traffic.
- **East-West:** Internal service-to-service traffic.

### Service Mesh Architecture

The document describes a sidecar-based service mesh.

```text
          Kubernetes Pod A
    +--------------------------+
    |                          |
    |  Application → Envoy     |
    |                 Sidecar  |
    +--------------------------+
                       |
                       | mTLS
                       |
                       v
          Kubernetes Pod B
    +--------------------------+
    |                          |
    |  Envoy → Application     |
    | Sidecar                  |
    +--------------------------+
```

The sidecar proxy manages network communication without requiring every application to implement the same infrastructure capabilities.

### Service Mesh Responsibilities

| Capability | Purpose |
|---|---|
| mTLS | Encrypt internal traffic and authenticate workloads |
| Authorization | Enforce service-to-service access policies |
| Retries | Handle transient failures |
| Circuit breaking | Protect against unhealthy dependencies |
| Load balancing | Distribute internal traffic |
| Distributed tracing | Track requests across services |
| Metrics | Monitor service communication |
| Traffic management | Support canary releases and traffic splitting |

### API Gateway vs. Service Mesh

| API Gateway | Service Mesh |
|---|---|
| External traffic | Internal traffic |
| Client authentication | Workload identity |
| API rate limiting | Service-level traffic controls |
| Public routing | Internal service discovery/routing |
| Edge security | Service-to-service security |

### Service Mesh Technologies

- Istio — Envoy-based service mesh.
- Linkerd — Lightweight service mesh.
- Consul Connect — Service connectivity and security.
- Cilium — eBPF-based networking and service mesh capabilities.

### Service Mesh Trade-offs

**Advantages:**
- Consistent security.
- Language-independent traffic policies.
- Centralized observability.
- Advanced traffic management.

**Costs:**
- Additional latency.
- CPU and memory overhead.
- More complex debugging.
- Control-plane operational burden.

The document suggests considering service meshes for large, polyglot microservice environments, especially where strict mTLS and advanced traffic management are required.

**Important:** The document's service-count thresholds are architectural heuristics, not strict adoption rules.

---

# 8. Retry, Exponential Backoff, and Jitter

### Problem: Retry Storms

Immediate retries can increase traffic to an already overloaded service.

```text
Service Fails
     |
     v
1,000 Clients Retry Immediately
     |
     v
Service Receives More Traffic
     |
     v
Further Overload
```

### Exponential Backoff

Retry delays increase after each failure.

```text
Attempt 1 → 100 ms
Attempt 2 → 200 ms
Attempt 3 → 400 ms
Attempt 4 → 800 ms
```

### Jitter

Add randomness to prevent all clients from retrying simultaneously.

```python
delay = base * (2 ** attempt) + random_jitter
```

**Critical rule:** Retried operations must be safe to repeat.

For example, payment requests should use idempotency keys to prevent duplicate charges.

---

# 9. Timeout Hierarchy

Timeouts should be coordinated across the request chain.

### Example

```text
Client Timeout       10s
       |
       v
Gateway Timeout       8s
       |
       v
Service A Timeout     6s
       |
       v
Service B Timeout     4s
       |
       v
Database Timeout      2s
```

The document recommends progressively shorter downstream timeouts.

**Why?**

If a caller times out before its downstream operation finishes, the downstream service may continue consuming resources unnecessarily.

**Key takeaway:** Configure timeouts as a coordinated hierarchy rather than independently.

---

# 10. Rate Limiting

Rate limiting prevents excessive traffic from overwhelming services.

### Common Algorithms

| Algorithm | Mechanism | Best use |
|---|---|---|
| Token Bucket | Tokens replenish at a fixed rate; bursts allowed | General APIs |
| Leaky Bucket | Smooths requests at a fixed rate | Traffic shaping |
| Fixed Window | Counts requests within fixed intervals | Simple limits |
| Sliding Window Log | Tracks request timestamps | Precise enforcement |
| Sliding Window Counter | Approximates sliding-window counts | Memory-efficient limits |

### Enforcement Layers

```text
Client
   |
   v
API Gateway
   |
   | Per-IP / Per-Client Limit
   v
Microservice
   |
   | Per-Tenant / Per-Feature Limit
   v
Database
   |
   | Connection Pool Limit
   v
Storage
```

**Architectural principle:** Rate limiting can be enforced at multiple layers depending on what resource needs protection.

---

# 11. Anti-Corruption Layer (ACL)

An Anti-Corruption Layer isolates internal domain models from external or legacy systems.

### Architecture

```text
Internal Application
         |
         v
Anti-Corruption Layer
         |
         +-- Data Transformation
         +-- Model Mapping
         +-- Retry / Circuit Breaker
         +-- Caching
         |
         v
Legacy / External System
```

### Responsibilities

- Translate external models into internal models.
- Prevent legacy assumptions from spreading into new services.
- Isolate external API changes.
- Apply resilience mechanisms at integration boundaries.

**Use case:** Integrating modern microservices with legacy enterprise systems or third-party APIs.

---

# 12. Strangler Fig Pattern

The Strangler Fig Pattern enables incremental migration from a monolith to microservices.

### Architecture

```text
                    Client
                       |
                       v
                 Routing Proxy
                       |
            +----------+----------+
            |                     |
            v                     v
      Legacy Monolith       New Microservices
            |                     |
       Existing APIs          Migrated APIs
```

### Migration Process

1. Place a routing layer in front of the monolith.
2. Identify functionality to migrate.
3. Implement that functionality in a new service.
4. Route selected traffic to the new service.
5. Gradually migrate remaining functionality.
6. Retire the monolith when appropriate.

**Benefits:**
- Incremental migration.
- Reduced deployment risk.
- Easier rollback through routing changes.
- No mandatory big-bang rewrite.

---

# 13. Observability in Distributed Systems

Microservices require visibility across service boundaries.

### Three Pillars of Observability

| Pillar | Purpose | Tools mentioned |
|---|---|---|
| Metrics | Quantitative system health | Prometheus, Datadog, CloudWatch |
| Logs | Individual application events | ELK, Splunk, Loki |
| Traces | End-to-end request execution | Jaeger, Zipkin, AWS X-Ray |

### RED Method — Service Monitoring

**R — Rate:** Requests per second.

**E — Errors:** Failed requests and error rates.

**D — Duration:** Request latency, including p50, p95, and p99.

### USE Method — Infrastructure Monitoring

**U — Utilization:** CPU, memory, disk usage.

**S — Saturation:** Resource contention and queue depth.

**E — Errors:** Resource-level failures.

### Distributed Tracing

```text
Trace ID: abc-123

Client Request
      |
      v
API Gateway         [Span 1]
      |
      v
Order Service       [Span 2]
      |
      v
Payment Service     [Span 3]
      |
      v
Fraud Service       [Span 4]
      |
      v
Database            [Span 5]
```

**Core concepts:**
- **Trace:** Complete request journey.
- **Span:** Individual operation.
- **Trace context:** Identifiers propagated across service boundaries.

Common propagation headers include:

```text
traceparent
x-request-id
```

**Key takeaway:** Distributed tracing helps identify which downstream dependency caused an upstream failure or latency increase.

---

# 14. Architecture Decision Framework

## Choosing Resilience Patterns

| Problem | Primary Pattern | Supporting Mechanism |
|---|---|---|
| Dependency repeatedly failing | Circuit Breaker | Fallback |
| Dependency extremely slow | Bulkhead | Latency-aware circuit breaker |
| Thread pool exhaustion | Bulkhead | Resource limits |
| Excessive incoming traffic | Rate Limiting | Backpressure |
| Retry storms | Exponential Backoff | Jitter |
| External API security | API Gateway | TLS, authentication |
| Different frontend requirements | BFF | Client-specific aggregation |
| Internal service security | Service Mesh | mTLS |
| Legacy system integration | Anti-Corruption Layer | Circuit breaker |
| Monolith migration | Strangler Fig | Routing proxy |
| Difficult distributed debugging | Distributed Tracing | Metrics and logs |

## When to Introduce Architectural Components

The document provides these approximate guidelines:

| System Scale | Suggested Architecture |
|---|---|
| 1–2 services | Direct calls, basic timeouts |
| 3–10 services | Circuit breakers, bulkheads, centralized logging |
| 10–50 services | API gateway, distributed tracing, application resilience |
| 50+ services | Consider service mesh, BFFs, advanced traffic management |

**Important:** Complexity, security requirements, traffic patterns, and team capabilities matter more than service count alone.

---

# 15. Key Architectural Takeaways

| # | Concept | Key takeaway |
|---|---|---|
| 1 | **Cascading failures** | Shared dependencies can propagate failures across otherwise healthy services. |
| 2 | **Synchronous dependencies** | Long chains increase aggregate failure exposure. |
| 3 | **Circuit breaker** | Stops repeated calls to unhealthy dependencies. |
| 4 | **Sliding window** | Evaluate recent failure rates instead of relying on naive counters. |
| 5 | **Latency-aware protection** | Slow dependencies can exhaust resources even without errors. |
| 6 | **Bulkhead** | Isolates resources and prevents one dependency from monopolizing capacity. |
| 7 | **Circuit breaker + bulkhead** | Combines failure detection with resource isolation. |
| 8 | **API Gateway** | Centralizes external security, routing, and traffic policies. |
| 9 | **BFF** | Optimizes backend interactions for specific client types. |
| 10 | **Service Mesh** | Provides internal traffic security, resilience, and observability. |
| 11 | **Exponential backoff + jitter** | Reduces synchronized retry storms. |
| 12 | **Timeout hierarchy** | Coordinates deadlines across dependent services. |
| 13 | **Rate limiting** | Prevents excessive requests from exhausting resources. |
| 14 | **Anti-Corruption Layer** | Protects internal domain models from external systems. |
| 15 | **Strangler Fig** | Supports incremental migration from monoliths. |
| 16 | **RED / USE monitoring** | Provides service-level and infrastructure-level health indicators. |
| 17 | **Distributed tracing** | Identifies the origin of failures across service boundaries. |

---

# 16. Final Mental Model: Production-Ready Microservices

The following combines the document's major patterns into one reference architecture.

```text
                         External Clients
                                |
                                v
                       +----------------+
                       |   API Gateway  |
                       |----------------|
                       | Authentication |
                       | Rate Limiting  |
                       | TLS / Routing  |
                       +----------------+
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
             Mobile BFF                   Web BFF
                  |                           |
                  +-------------+-------------+
                                |
                                v
                        Order Service
                                |
                     +----------+----------+
                     |                     |
                     v                     v
                 Bulkhead A            Bulkhead B
                     |                     |
                     v                     v
              Circuit Breaker A     Circuit Breaker B
                     |                     |
                     v                     v
               Payment Service       Inventory Service
                     |                     |
                     v                     v
                 Database               Database

            <------ Service Mesh ------>

                  Internal mTLS
                  Traffic Policies
                  Retries / Timeouts
                  Load Balancing
                  Distributed Tracing

                           |
                           v
                 Observability Platform
                 ----------------------
                 Metrics (RED / USE)
                 Centralized Logs
                 Distributed Traces
                 Alerts / Dashboards
```

### How the Architecture Handles Failures

**Scenario 1 — Payment service becomes unavailable**

The circuit breaker detects failures and opens. Subsequent calls fail fast instead of continuously consuming resources.

**Scenario 2 — Inventory service becomes extremely slow**

The bulkhead limits concurrent inventory calls, preventing the inventory dependency from exhausting resources needed by payment processing.

**Scenario 3 — Sudden traffic spike**

The API Gateway applies rate limits to protect downstream services.

**Scenario 4 — Internal communication security**

The service mesh provides mTLS and workload-level traffic policies.

**Scenario 5 — Request latency increases**

Distributed tracing identifies the slow dependency, while RED and USE metrics help determine whether the cause is service errors, saturation, or resource exhaustion.

---

## Final Architectural Principle

**A resilient microservices architecture is not one where services never fail. It is one where individual failures remain isolated, recovery is controlled, and the overall system continues operating wherever possible.**

The document's central design approach can be summarized as:

**Prevent overload → Isolate resources → Detect failures → Fail fast or degrade gracefully → Recover safely → Observe continuously.**

This condensed version covers the document's main technical sections, preserving its architectural patterns, operational mechanisms, and engineering trade-offs. 8_Microservices_Patterns
