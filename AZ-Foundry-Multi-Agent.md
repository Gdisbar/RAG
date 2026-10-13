# Multi-Agent Orchestration in Azure AI Foundry — A Hands-On Tutorial (for LangGraph devs)

You already think in graphs: nodes, edges, state, conditional routing. Azure AI Foundry gives you three different layers for the same job, at increasing levels of control. This tutorial builds all three using the exact scenario from your Modern Workplace Assistant — split into a **Policy specialist**, a **Tech Docs specialist**, and an **Orchestrator** — so you can see the same problem solved three ways.

**⚠️ Preview-software warning:** almost everything below is public preview and moves fast (we hit two breaking surprises just fixing your single-agent script earlier this week). Treat every constructor signature here as "probably right, verify before you ship" — see the Gotchas section at the end for how.

---

## The mental model, up front

| LangGraph concept | Azure AI Foundry equivalent |
|---|---|
| `StateGraph` | `agent_framework.WorkflowBuilder` |
| A node (plain function) | An `Executor` subclass with an `@handler` method, or an agent auto-wrapped as one |
| `add_node` + `add_edge` | `.add_edge(source, target)` |
| `add_conditional_edges` | `.add_edge(source, target, condition=callable)` |
| `Send()` / parallel branches | `.add_fan_out_edge(source, [targets])` |
| A join/reducer step | `.add_fan_in_edge([sources], target)` |
| Hand-rolled supervisor pattern | **Connected Agents** (`ConnectedAgentTool`) — zero orchestration code |
| Hand-rolled plan-and-execute / ReAct loop | **Magentic** orchestration (a manager that plans, delegates, and re-plans) |
| Checkpointer / persistence | Foundry conversations/threads; for full workflows, a hosted **Workflow agent** |
| Deploying your graph as a service | `WorkflowAgentDefinition` (CSDL) hosted directly in Foundry |

Three layers, increasing power and code:

1. **Connected Agents** — a supervisor agent delegates to specialist agents purely by natural language. No orchestration code at all.
2. **Microsoft Agent Framework Workflows** — you build the graph yourself (`WorkflowBuilder`, executors, edges). This is the direct LangGraph equivalent.
3. **Prebuilt orchestration patterns** (Sequential, Concurrent, Handoff, Group Chat, Magentic) — pre-built graphs for common shapes, so you don't wire edges by hand.

---

## Prerequisites

```bash
pip install azure-ai-projects azure-identity --break-system-packages
pip install agent-framework --pre --break-system-packages   # for Part 2 & 3 only
az login
```

Same environment variables as before:

```
FOUNDRY_PROJECT_ENDPOINT=...
FOUNDRY_MODEL_NAME=...
STORAGE_CONNECTION_STRING=...
CONTAINER_NAME=...
MCP_SERVER_URL=...
```

---

## Part 1 — Connected Agents: the "just delegate" pattern

This is the lowest-code option. You create two specialist agents exactly like you already know how to (`PromptAgentDefinition` + tools), then give a third "orchestrator" agent a `ConnectedAgentTool` pointing at each one. Foundry handles the agent-to-agent round trip for you server-side — unlike `FunctionTool`, there's no manual loop to write.

```python
import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    MCPTool,
    FunctionTool,
    ConnectedAgentTool,
)
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
)

# --- reuse the blob tools from your existing script ---
blob_list_tool = FunctionTool(
    name="get_blob_list",
    description="List company documents available in Blob Storage.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    strict=True,
)
blob_document_tool = FunctionTool(
    name="get_company_document",
    description="Retrieve the full text of a company document by file name.",
    parameters={
        "type": "object",
        "properties": {"blob_name": {"type": "string"}},
        "required": ["blob_name"],
        "additionalProperties": False,
    },
    strict=True,
)
mcp_tool = MCPTool(
    server_url=os.environ["MCP_SERVER_URL"],
    server_label="Microsoft_Learn_Documentation",
    require_approval="always",
)

# --- Specialist 1: Policy ---
policy_agent = project_client.agents.create_version(
    agent_name="Policy-Agent",
    definition=PromptAgentDefinition(
        model=os.environ["FOUNDRY_MODEL_NAME"],
        instructions=(
            "You answer questions about Contoso's internal policies using "
            "get_blob_list and get_company_document. Always cite the file name."
        ),
        tools=[blob_list_tool, blob_document_tool],
    ),
)

# --- Specialist 2: Technical docs ---
techdocs_agent = project_client.agents.create_version(
    agent_name="TechDocs-Agent",
    definition=PromptAgentDefinition(
        model=os.environ["FOUNDRY_MODEL_NAME"],
        instructions="You answer Azure/M365 technical questions using Microsoft Learn, with links.",
        tools=[mcp_tool],
    ),
)

# --- Orchestrator: delegates to both specialists ---
# NOTE: ConnectedAgentTool just landed natively in azure.ai.projects.models
# (SDK 2.3.0, July 2026). Its shape has long mirrored
# azure.ai.agents.models.ConnectedAgentTool(id, name, description) -- sanity
# check it before depending on it:
#   import inspect; print(inspect.signature(ConnectedAgentTool.__init__))
orchestrator = project_client.agents.create_version(
    agent_name="Workplace-Orchestrator",
    definition=PromptAgentDefinition(
        model=os.environ["FOUNDRY_MODEL_NAME"],
        instructions=(
            "Route policy questions to the policy specialist and technical "
            "questions to the tech docs specialist. For combined questions, "
            "call both and merge their answers into one response."
        ),
        tools=[
            ConnectedAgentTool(
                id=policy_agent.id,
                name=policy_agent.name,
                description="Answers questions about Contoso's internal policies.",
            ),
            ConnectedAgentTool(
                id=techdocs_agent.id,
                name=techdocs_agent.name,
                description="Answers Azure/Microsoft 365 technical implementation questions.",
            ),
        ],
    ),
)
```

Calling it is identical to what you already have — same `responses.create(... extra_body={"agent_reference": {"name": orchestrator.name, ...}})` pattern, no new client-side round-trip logic needed for the delegation itself.

**When to use this:** you want quick, reusable specialist agents and simple routing, without writing or maintaining any graph code. **Limit to know:** a connected agent can't itself have connected sub-agents — one level of delegation only.

---

## Part 2 — Microsoft Agent Framework Workflows: the real LangGraph equivalent

This is the direct analog to `StateGraph`. You define `Executor` nodes, wire them with `.add_edge()` / `.add_fan_out_edge()` / `.add_fan_in_edge()`, and the framework routes messages between them — including agents backed by your Foundry project.

```python
import asyncio
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

credential = AzureCliCredential()
# Reads FOUNDRY_PROJECT_ENDPOINT / FOUNDRY_MODEL from the environment
chat_client = FoundryChatClient(credential=credential)

policy_agent = chat_client.as_agent(
    name="PolicyAgent",
    instructions="Answer Contoso policy questions. Cite the file name.",
)
techdocs_agent = chat_client.as_agent(
    name="TechDocsAgent",
    instructions="Answer Azure/M365 technical questions with official guidance.",
)


class Dispatcher(Executor):
    """Fan-out entry point: forwards the incoming question to both specialists."""

    @handler
    async def dispatch(self, question: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(question)


class Aggregator(Executor):
    """Fan-in target: waits for both specialist answers, then merges them."""

    @handler
    async def aggregate(self, answers: list, ctx: WorkflowContext) -> None:
        merged = "\n\n---\n\n".join(str(a) for a in answers)
        await ctx.yield_output(merged)


dispatcher = Dispatcher(id="dispatcher")
aggregator = Aggregator(id="aggregator")

# Diamond shape: dispatcher fans out to both specialists, both fan in to aggregator
workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_fan_out_edge(dispatcher, [policy_agent, techdocs_agent])
    .add_fan_in_edge([policy_agent, techdocs_agent], aggregator)
    .build()
)


async def main():
    result = await workflow.run(
        "What is Contoso's remote work policy, and how do I configure "
        "Conditional Access in Azure to enforce it?"
    )
    print(result.get_outputs())


asyncio.run(main())
```

Notes:
- Passing `policy_agent` / `techdocs_agent` (returned by `as_agent`) straight into `add_fan_out_edge` works — the framework auto-wraps agents as executors for you.
- Want conditional routing instead of parallel fan-out (e.g., only call TechDocs if the question mentions Azure)? Pass `condition=your_callable` to `.add_edge(...)` — same shape as LangGraph's conditional edges.
- Want a sequential pipeline instead (policy answer feeds into tech docs as context)? Swap the fan-out/fan-in for a plain chain: `.add_edge(policy_agent, techdocs_agent)`.

**When to use this:** you need custom routing logic, conditional branches, or a topology the prebuilt patterns don't cover — same reason you'd reach for LangGraph over a canned agent loop.

---

## Part 3 — Prebuilt orchestration patterns

As of the `agent-framework-orchestrations` package reaching 1.0, five common shapes ship as ready-made builders so you don't wire edges by hand:

| Pattern | Shape | Use it when |
|---|---|---|
| **Sequential** | Pipeline, each agent's output feeds the next | Deterministic multi-step pipelines (draft → review → finalize) |
| **Concurrent** | Fan-out/fan-in, same input to all | Independent parallel research/analysis that gets merged |
| **Handoff** | Dynamic routing between agents based on conversation | Support-style triage where the "right" agent changes mid-conversation |
| **Group Chat** | Multiple agents converse in shared context | Debate/review loops (e.g. your reviewer-until-approved example) |
| **Magentic** | A manager plans, assigns work, monitors progress, re-plans on stalls | Open-ended tasks where you don't know the solution path in advance |

Magentic is the closest thing to "give it a goal and let it figure out the routing" — the best match for LangGraph's plan-and-execute or ReAct-style graphs:

```python
# Package/import path has shifted around the 1.0 release -- check the current
# samples directory in microsoft/agent-framework if this import doesn't resolve.
from agent_framework_orchestrations import MagenticBuilder

workflow = (
    MagenticBuilder()
    .participants(policy=policy_agent, techdocs=techdocs_agent)
    .with_standard_manager(
        chat_client=chat_client,
        max_round_count=10,
        max_stall_count=3,
        max_reset_count=2,
    )
    .build()
)

result = await workflow.run(
    "Give me a full compliance-and-implementation plan for enforcing our "
    "remote work security policy in Azure, with links to Microsoft docs."
)
```

The manager decides, round by round, whether to ask `policy` or `techdocs` next, tracks whether it's making progress, and re-plans automatically if it stalls — instead of you hand-coding that supervisor logic in a LangGraph node.

**When to use this over hand-built Workflows:** the task shape matches one of the five patterns and you'd rather not maintain custom graph code for something this common.

---

## Part 4 — Going to production: hosted Workflow agents

Everything in Parts 2–3 runs on your machine, calling out to Foundry-hosted models. For a persistent, versioned, server-managed equivalent of your graph — closer to deploying a compiled LangGraph app as a service — Foundry has a native `WorkflowAgentDefinition`:

```python
from azure.ai.projects.models import WorkflowAgentDefinition

workflow_agent = project_client.agents.create_version(
    agent_name="Workplace-Workflow",
    definition=WorkflowAgentDefinition(
        csdl="<compiled workflow definition>",  # see below
    ),
)
```

The `csdl` (the workflow's declarative schema) is exactly what Agent Framework's `WorkflowBuilder` compiles to under the hood. The practical path: prototype your graph locally with `WorkflowBuilder` as in Part 2, then use Agent Framework's export tooling to get the declarative form and host it as a `WorkflowAgentDefinition`. There's a full worked example of this "local workflow → hosted Foundry agent" path in `microsoft/AIFoundry-AgentsV2-HostedWorkflow` on GitHub — worth reading end-to-end before you build this in production, since the export step is the part most likely to have moved since this was written.

---

## Gotchas learned the hard way (some from fixing your script this week)

1. **Preview SDK signatures shift between releases.** `FunctionTool` itself changed shape — an early preview took `functions=[callable, ...]` (auto-introspected), the version you're on takes explicit `name=`/`description=`/`parameters=`/`strict=`. Before trusting any tutorial's exact kwargs — including this one — check what's actually installed:
   ```python
   import inspect
   from azure.ai.projects.models import ConnectedAgentTool
   print(inspect.signature(ConnectedAgentTool.__init__))
   ```
2. **Generation params live on the agent, not the call.** `reasoning`, `temperature`, `top_p` must be set in `PromptAgentDefinition` at creation time. Passing them to `responses.create()` once you're using `agent_reference` gets rejected with `"Not allowed when agent is specified"` — you hit this exact error already.
3. **Always check the real `response.status`.** A call can return with no exception and still have produced nothing usable — small "nano"/"mini" reasoning models can burn their whole token budget on internal reasoning before writing a visible answer. Don't assume success just because nothing threw.
4. **Not all tools auto-execute.** Plain `FunctionTool` calls require you to run the function yourself and submit `function_call_output` (the manual loop you already built). `ConnectedAgentTool` and MCP tools push more of that round-trip onto Foundry's side — MCP still needs approval handling, Connected Agents is fully server-side.
5. **Connected agents are one level deep.** A sub-agent can't itself have connected sub-agents — you'll get a tool-call-depth error if you try.

---

## Where to go deeper

- Connected agents how-to: `learn.microsoft.com/azure/ai-foundry/agents/how-to/connected-agents`
- Agent Framework workflows docs: `learn.microsoft.com/agent-framework/workflows`
- Orchestration patterns (Sequential/Concurrent/Handoff/GroupChat/Magentic): `learn.microsoft.com/agent-framework/workflows/orchestrations`
- Hosted workflow example repo: `github.com/microsoft/AIFoundry-AgentsV2-HostedWorkflow`
- Agent Framework samples (all patterns, Python + .NET): `github.com/microsoft/agent-framework`
