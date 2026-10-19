# API Architecture: REST, GraphQL, gRPC, and Beyond


**Audience:** Senior engineers, staff engineers, and system architects  
**Focus:** API contracts, architectural patterns, operational mechanisms, and engineering trade-offs

---

## 1. The Network Boundary

A local function call becomes a contract between independently evolving systems once it crosses a network. Network calls introduce variable latency, serialization overhead, partial failure, independent memory spaces, and runtime schema compatibility concerns.

| Local call | Networked call |
|---|---|
| Predictable latency | Variable latency and timeouts |
| No network serialization | Serialization/deserialization cost |
| Shared process and memory | Independent failure domains |
| Compiler-enforced local types | Runtime contract and schema enforcement |
| Simple synchronous execution | Sync/async behavior and retries |

The eight fallacies of distributed computing are useful reminders: networks are not perfectly reliable, latency is not zero, bandwidth is not infinite, networks are not inherently secure, topology can change, administration is not always unified, transport has a cost, and environments are not homogeneous.

**Design principle:** Specify failure behavior and client-visible guarantees, not just endpoint syntax.

## 2. Three Foundational API Design Questions

API styles differ along three axes:

1. **Addressing:** How is an operation identified?
2. **Serialization:** How is data represented on the wire?
3. **Flow control:** Is communication request/response, streaming, or both?

| Style | Addressing | Data format | Flow |
|---|---|---|---|
| REST | Resource URLs + HTTP verbs | Usually JSON | Request/response |
| GraphQL | Query document at an endpoint | JSON | Request/response; subscriptions vary by implementation |
| gRPC | Service/method defined in `.proto` | Protobuf by default | Unary and streaming RPCs |
| tRPC | Typed TypeScript procedures | Commonly JSON transport | Request/response |
| WebSocket | URL + application message types | Text or binary | Full-duplex |

## 3. REST: Resource-Oriented APIs

REST models domain resources and uses HTTP semantics as part of the contract.

### 3.1 HTTP methods and semantics

| Method | Typical meaning | Idempotent? | Safe? |
|---|---|---:|---:|
| `GET` | Read resource | Yes | Yes |
| `POST` | Create or execute an operation | Usually no | No |
| `PUT` | Replace resource representation | Yes | No |
| `PATCH` | Partially modify resource | Depends on implementation | No |
| `DELETE` | Delete resource | Yes in HTTP semantics | No |

HTTP semantics enable standard infrastructure behavior: `GET` caching, intermediary handling, and clearer retry decisions. Idempotency is especially important when a client retries after a timeout and cannot tell whether the server completed the operation.

### 3.2 Overfetching and underfetching

- **Overfetching:** A response contains fields the client does not need.
- **Underfetching:** The client must make multiple calls to assemble one screen or workflow.

Example of sequential resource requests:

```text
GET /users/123
        |
        +--> GET /users/123/orders
        |
        +--> GET /users/123/payment-methods
```

Mitigations include a Backend-for-Frontend (BFF), field-selection query parameters, and resource expansion. These can reduce round trips but may add endpoint variants, payload variability, or caching complexity.

### 3.3 Model domain actions explicitly

Not every operation is CRUD. Canceling an order may trigger refunds, inventory release, warehouse actions, notifications, authorization, and audit logging. Deleting the order or merely patching its status hides those domain semantics.

```http
POST /orders/123/cancel
Content-Type: application/json

{
  "reason": "customer_request",
  "refund_method": "original_payment"
}
```

Use action sub-resources for meaningful state transitions such as `cancel`, `approve`, `publish`, `archive`, `merge`, or `transfer`. This creates an explicit domain boundary for validation, authorization, audit, and side-effect orchestration.

### 3.4 Standardized errors

RFC 7807 Problem Details provides a structured error representation, including `type`, `title`, `status`, `detail`, and `instance`, with optional domain-specific fields.

- `type` is a stable machine-readable error identifier.
- Human-readable wording can change without forcing clients to parse strings.
- The API designer must still choose appropriate HTTP status codes, such as `400`, `403`, `409`, or `422`.

### 3.5 REST maturity and hypermedia

| Level | Pattern |
|---|---|
| 0 | One endpoint and usually one verb |
| 1 | Resource-specific URLs |
| 2 | HTTP verbs and status codes used semantically |
| 3 | Hypermedia controls (HATEOAS) guide available actions |

HATEOAS returns links to permitted actions based on resource state. It is architecturally expressive, but dynamic links can add backend work and fit awkwardly with clients built around static routes and generated API clients.

**Trade-off:** Level 2 REST plus OpenAPI and versioning is often a pragmatic choice; HATEOAS remains useful for APIs where dynamic capability discovery is valuable.

### 3.6 Pagination: offset vs. cursor

**Offset pagination** supports direct page jumps but can become expensive at large offsets and can produce duplicates or omissions when records change between requests.

```sql
SELECT *
FROM orders
WHERE user_id = 123
ORDER BY created_at DESC
LIMIT 20 OFFSET 100000;
```

**Cursor/keyset pagination** uses the last-seen ordering key to continue from a stable position.

```sql
SELECT *
FROM orders
WHERE user_id = 123
  AND created_at < :cursor
ORDER BY created_at DESC
LIMIT 20;
```

| Offset | Cursor/keyset |
|---|---|
| Supports page-number navigation | Optimized for sequential navigation |
| Large offsets can be expensive | Can seek through an appropriate index |
| Susceptible to page drift during inserts/deletes | More stable for feeds and timelines |
| Useful for small datasets and page jumps | Useful for high-volume lists and infinite scroll |

Use a deterministic ordering key; when timestamps can collide, include a unique tie-breaker such as `id`.

## 4. GraphQL: Client-Defined Queries

GraphQL lets clients specify the fields and relationships they need in a query. It can reduce overfetching and the number of network round trips, and its schema provides a typed contract.

```graphql
query GetUserWithOrders {
  user(id: "123") {
    name
    avatar
    orders(limit: 5) {
      id
      total
      items {
        name
        price
      }
    }
  }
}
```

### 4.1 Caching and query governance

A common GraphQL deployment uses a single endpoint, which makes ordinary URL-based HTTP caching less straightforward than resource-oriented REST.

Mitigations include:

- **Persisted queries:** Clients send a known query identifier or hash; approved queries can use cache-friendly transport where appropriate.
- **Application-layer caching:** Cache resolver results or complete responses with carefully designed keys and invalidation.
- **Query complexity/depth limits:** Reject expensive or excessively nested queries before they overload downstream systems.
- **Rate and resource limits:** Apply cost-aware limits, not only request counts.

Persisted queries improve control and can reduce query text overhead, but do not automatically make every GraphQL response safely cacheable.

### 4.2 N+1 database query problem

GraphQL can collapse many HTTP requests into one while still causing many database queries inside the server.

```text
GraphQL request
      |
      v
Resolve user                1 DB query
      |
      v
Resolve 50 friends           1 DB query
      |
      +--> latest post for friend 1
      +--> latest post for friend 2
      +--> ...
      +--> latest post for friend 50
                             50 DB queries
```

A single request can therefore produce dozens of database calls.

**DataLoader** batches and deduplicates loads within a request execution window, then maps results back to their original keys.

Implementation requirements:
- Create loaders per request when their cache is request-scoped.
- Return results in the same order as the input keys.
- Handle missing keys and cache invalidation.
- Measure query count and execution cost; batching is not automatic for arbitrary resolver logic.

### 4.3 Federation and schema ownership

GraphQL Federation and schema stitching allow separate teams or services to own parts of a graph while exposing a unified schema.

```text
Client
  |
  v
GraphQL Gateway / Router
  |                 |
  v                 v
User subgraph    Order subgraph
  |                 |
  v                 v
User data        Order data
```

**Benefits:** Domain ownership and a unified client-facing query model.  
**Costs:** Gateway planning and routing, cross-service latency, schema governance, and harder debugging across subgraphs.

## 5. gRPC: Typed, Efficient RPC

gRPC typically uses Protocol Buffers (protobuf) for schema-first, binary serialization and HTTP/2 for transport. Service definitions generate client/server code and make interface contracts explicit.

```protobuf
syntax = "proto3";

service OrderService {
  rpc GetOrder(GetOrderRequest) returns (Order);
  rpc StreamOrders(StreamOrdersRequest) returns (stream Order);
}

message GetOrderRequest {
  string order_id = 1;
}

message Order {
  string id = 1;
  string customer_id = 2;
  double total = 3;
}
```

### 5.1 Protobuf and schema evolution

- Binary encoding is often smaller and faster to parse than verbose JSON, though actual gains depend on payloads and workload.
- Numeric field tags avoid repeating field names on the wire.
- Field numbers are part of the compatibility contract: do not reuse removed field numbers; reserve them when appropriate.
- Schema changes still require compatibility discipline across deployed clients and servers.

### 5.2 HTTP/2 and RPC patterns

HTTP/2 supports multiplexed streams over a connection and header compression. It reduces per-request connection overhead, but TCP packet loss can still block progress for multiple streams on that TCP connection.

gRPC supports four RPC shapes:

1. **Unary:** One request, one response.
2. **Server streaming:** One request, multiple responses.
3. **Client streaming:** Multiple requests, one response.
4. **Bidirectional streaming:** Both sides send streams independently.

Streaming is useful for telemetry, log ingestion, live updates, and long-running data exchange. It also introduces flow-control, cancellation, deadlines, and stream-lifecycle concerns.

### 5.3 Browser interoperability

Browsers do not expose all the HTTP/2 features used by native gRPC, including the required trailer behavior. Common options are:

| Option | Mechanism | Trade-off |
|---|---|---|
| gRPC-Web | Browser-compatible protocol translated by a proxy | Extra component/hop and feature constraints |
| Connect-RPC | HTTP-compatible RPC protocol with binary and JSON options | Different ecosystem and protocol choices |
| Twirp | RPC over standard HTTP/1.1 using JSON or protobuf | Simpler model; fewer streaming capabilities |
| Envoy | Proxy can translate and route supported traffic | Operational configuration and maintenance |

**Typical boundary choice:** gRPC for controlled service-to-service communication; REST or GraphQL for many public/browser-facing interfaces.

## 6. Gateways, Service Meshes, and Protocol Translation

### 6.1 API gateway

A gateway can expose REST/JSON externally while routing to internal gRPC services.

```text
External client
  REST / JSON
      |
      v
+-----------------------------+
| API Gateway                 |
| Auth, validation, limits,   |
| logging, routing, translation|
+-----------------------------+
      |
      | gRPC / Protobuf
      v
Internal service
```

**Benefits**
- Separates public contracts from internal service interfaces.
- Centralizes authentication, authorization, rate limiting, validation, and observability.
- Allows internal teams to evolve independently within compatibility rules.

**Costs**
- Translation and routing add latency.
- The gateway becomes a critical dependency and potential blast-radius amplifier.
- Requires horizontal scaling, health checks, monitoring, and controlled deployment.

Resilience controls include circuit breakers, bulkheads, rate limiting, deadlines/timeouts, and bounded retries. Retries must account for idempotency and should not amplify overload.

### 6.2 Service mesh

A service mesh moves many service-to-service traffic controls into infrastructure proxies.

```text
Service A
   |
   v
Sidecar / node proxy
   |  mTLS, policy, telemetry,
   |  retries, timeouts, traffic split
   v
Sidecar / node proxy
   |
   v
Service B
```

Examples include Istio, Linkerd, and Consul Connect.

**Benefits:** Consistent transport security, traffic policy, telemetry, and rollout controls without duplicating all logic in every service.  
**Costs:** More infrastructure, configuration surface, debugging complexity, and platform-team responsibility.

A mesh complements rather than replaces application-level authorization, correct retry semantics, and business logic.

## 7. Modern API and Event Interfaces

### 7.1 tRPC

tRPC offers end-to-end TypeScript type safety with low schema/code-generation overhead.

**Best fit:** Full-stack TypeScript teams and monorepos.  
**Limitations:** Less suitable for polyglot clients, independently versioned public APIs, or ecosystems that need language-neutral contracts.

### 7.2 WebSockets vs. Server-Sent Events (SSE)

| Feature | WebSockets | SSE |
|---|---|---|
| Direction | Bidirectional | Server to client |
| Data | Text or binary | Text event stream |
| Reconnection | Usually application-managed | Built-in browser reconnection support |
| Common uses | Chat, gaming, collaborative editing | Notifications, dashboards, log streams |

Choose based on communication direction, delivery/reconnection needs, and infrastructure constraints.

### 7.3 HTTP/3 and QUIC

HTTP/3 uses QUIC over UDP. It supports faster connection establishment, avoids TCP-level head-of-line blocking between independent streams, and can migrate connections across network changes. Packet loss can still affect the relevant QUIC stream, and deployment behavior depends on client, server, and intermediary support.

HTTP/3 does not automatically make every RPC or API faster; measure the actual workload and network conditions.

### 7.4 AsyncAPI

OpenAPI describes HTTP APIs; AsyncAPI describes event-driven and message-based contracts such as Kafka, RabbitMQ, and supported WebSocket interfaces.

Asynchronous contracts should specify:
- Message schemas and versioning.
- Channel/topic names and direction.
- Delivery and ordering assumptions.
- Correlation identifiers and error handling.
- Consumer compatibility and replay behavior.

Event-driven systems need explicit contracts just as synchronous APIs do.

## 8. Architectural Decision Framework

| Requirement | Strong starting point | Main consideration |
|---|---|---|
| Public, resource-oriented API | REST + OpenAPI | HTTP semantics, caching, tooling |
| Client-specific nested data needs | GraphQL | Query governance, caching, N+1 prevention |
| High-throughput internal RPC | gRPC + protobuf | Schema compatibility, deadlines, streaming |
| Full-stack TypeScript monorepo | tRPC | Language and deployment coupling |
| Bidirectional real-time interaction | WebSockets | Connection lifecycle and backpressure |
| Server-to-browser event stream | SSE | One-way communication and reconnection |
| Durable asynchronous integration | Broker + AsyncAPI | Delivery semantics, ordering, idempotency |

### Common hybrid architecture

```text
                    Clients
                       |
            +----------+----------+
            |                     |
            v                     v
       REST / BFF             GraphQL API
            |                     |
            +----------+----------+
                       |
                API Gateway / Edge
                       |
        +--------------+---------------+
        |                              |
        v                              v
 Internal synchronous calls      Async events
        |                              |
   gRPC / protobuf                 Kafka / broker
        |                              |
        v                              v
  Service mesh                 Consumers / workflows
```

A hybrid model uses each protocol where its semantics fit: REST for interoperable public resources, GraphQL for client-shaped aggregation, gRPC for typed internal RPC, and messaging for decoupled asynchronous propagation.

## 9. Common Simplifications to Avoid

| Simplification | More accurate view |
|---|---|
| "REST is just CRUD over HTTP." | REST can model domain actions and state transitions; resource semantics matter. |
| "GraphQL replaces REST." | GraphQL addresses client-shaped data needs but adds caching, query-cost, and resolver concerns. |
| "gRPC is always faster." | Performance depends on payload, network, proxy hops, workload, and implementation. |
| "Gateways only translate protocols." | Gateways also enforce policy and become critical operational dependencies. |
| "HATEOAS is dead." | It is uncommon in some SPA architectures but remains useful where dynamic capability discovery matters. |
| "Cursor pagination is always better." | It excels at sequential high-volume access; offsets can suit page-number navigation and small datasets. |
| "Binary protocols are more secure." | Encoding is not security. Use authentication, authorization, and TLS/mTLS. |
| "A service mesh replaces application resilience." | It provides traffic controls, not business-level correctness or safe retry semantics. |

## 10. Operational Checklist

- [ ] Define API contracts and compatibility rules before implementation.
- [ ] Specify timeouts/deadlines, cancellation, retry budgets, and idempotency behavior.
- [ ] Use authentication and authorization at the correct boundaries; encrypt traffic with TLS/mTLS as appropriate.
- [ ] Apply rate limits and query/resource budgets to protect downstream services.
- [ ] Instrument every boundary with structured logs, metrics, and distributed tracing (for example, OpenTelemetry).
- [ ] Test partial failures, timeouts, duplicate requests, schema evolution, and overload.
- [ ] Monitor gateway/mesh health and account for their failure domains.
- [ ] Version and validate OpenAPI, GraphQL, protobuf, and asynchronous message contracts in CI.
- [ ] Measure serialization, query count, cache hit rate, and translation overhead rather than assuming a protocol is faster.
- [ ] Document ownership, deprecation, and migration paths for clients and services.

## 11. Architect's Takeaways

1. **Design for failure:** Every network call can time out, fail partially, or return after the client has stopped waiting.
2. **Choose the weakest sufficient abstraction:** Avoid GraphQL for simple resources or gRPC for a public browser interface unless the requirements justify it.
3. **Treat schemas as contracts:** OpenAPI, GraphQL SDL, protobuf, and AsyncAPI require compatibility and versioning discipline.
4. **Cache at the right layer:** REST benefits from HTTP caching; GraphQL often needs application-aware caching; RPC caching is usually explicit.
5. **Make observability cross-cutting:** Trace requests and events across protocol boundaries.
6. **Budget for translation:** Gateways and proxies add latency, operational dependencies, and failure modes.
7. **Prefer hybrid architectures when justified:** Public APIs, internal RPC, and asynchronous integration have different requirements.

**Bottom line:** There is no universally best API style. Choose based on client types, domain semantics, performance, caching, streaming, compatibility, team skills, and operational capacity.

## References and Further Reading

1. Roy Fielding (2000), *Architectural Styles and the Design of Network-based Software Architectures*.
2. Richardson Maturity Model.
3. RFC 7231, HTTP/1.1 Semantics and Content.
4. RFC 7807, Problem Details for HTTP APIs.
5. GraphQL Specification: https://graphql.github.io/graphql-spec/
6. gRPC Documentation: https://grpc.io/
7. Martin Kleppmann (2017), *Designing Data-Intensive Applications*.
8. Sam Newman (2021), *Building Microservices*, 2nd Edition.
9. Connect-RPC: https://connectrpc.com/
10. AsyncAPI: https://www.asyncapi.com/
11. IETF QUIC and HTTP/3 specifications.
