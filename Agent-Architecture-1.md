Here are the rendered ASCII diagrams for the three Mermaid diagrams found in the document:
1. High-Level Architecture Flowchart
┌──────────────────────────────────────────────────────────┐
│              Ticket and knowledge sources               │
│                                                          │
│   ┌──────────┐         ┌───────────┐      ┌───────────┐  │
│   │   Jira   │         │ ServiceNow│      │Confluence │  │
│   └────┬─────┘         └─────┬─────┘      └─────┬─────┘  │
└────────┼─────────────────────┼──────────────────┼────────┘
         │                     │                  │ (approved spaces)
         ▼                     ▼                  ▼
┌──────────────────────────────────────────────┬───────────┐
│              Ticket Poller                   │ Indexer   │
│              (Scheduled Job)                 └─────┬─────┘
└──────────────────────┬─────────────────────────────┤      
                       │                             │      
                       ▼                             ▼      
┌──────────────────────────────────────────────┬───────────┐
│                  Service Bus                 │ AI Search │
│                (Ticket Queue)                │(SOP Index)│
└──────────────────────┬───────────────────────┴─────▲─────┘
                       │                             │      
                       ▼                             │      
┌──────────────────────────────────────────────┐     │      
│            LangGraph Orchestrator            │     │      
│   (Categorise, RCA, Propose, Critic, Judge)  │     │      
└────────┬─────────────┬───────────────┬───────┘     │      
         │             │               │             │      
         ▼             ▼               ▼             │      
  ┌────────────┐ ┌──────────┐  ┌──────────────┐     │      
  │  Postgres  │ │Azure AOAI│  │ Tool Gateway │─────┘      
  └────────────┘ └──────────┘  └───────┬──────┘            
         │                             │                    
         ▼                             ├─────────┐          
  ┌────────────┐                       │         ▼          
  │ Approval   │                       │   ┌───────────┐    
  │  API / UI  │                       │   │ Target    │    
  └──────┬─────┘                       │   │ Servers   │    
         │                             │   └─────▲─────┘    
         ▼                             │         │          
  ┌────────────┐                       │         │ (Elevated
  │   Human    │                       │         │  Execution)
  │  Approver  │                       │         │          
  └──────┬─────┘                       │         │          
         │                             │         │          
         ▼                             │         │          
  ┌────────────┐                       │         │          
  │  Executor  │───────────────────────┴─────────┘          
  │   Worker   │ (Approved Plan Execution)                  
  └────────────┘                                            

(Source)
2. Agent Workflow (LangGraph State Diagram)
       ┌──────────┐
       │ [*] Start│
       └────┬─────┘
            │
            ▼
       ┌──────────┐
  ┌───>│  Intake  │
  │    └────┬─────┘
  │         │
  │         ▼
  │    ┌──────────┐    Out-of-scope / Low Confidence
  │    │Categorise│─────────────────────────────────┐
  │    └────┬─────┘                                 │
  │         │ In-scope category                     │
  │         ▼                                       │
  │    ┌──────────┐          No relevant SOP        │
  │    │ Retrieve │─────────────────────────────────┤
  │    └────┬─────┘                                 │
  │         │ SOPs found                            │
  │         ▼                                       │
  │  ┌──────────────┐                               │
  │  │   Diagnose   │<───────────┐                  │
  │  └──────┬───────┘            │                  │
  │         │                    │ Read-only        │
  │         ▼                    │ Evidence         │
  │    ┌──────────┐              │                  │
  │    │   RCA    │──────────────┘                  │
  │    └────┬─────┘  More evidence needed           │
  │         │ Root cause supported                  │
  │         ▼                                       │
  │    ┌──────────┐                                 │
  │    │ Propose  │<───────────┐                    │
  │    └────┬─────┘            │                    │
  │         │                  │ Rejected/Revise    │
  │         ▼                  │                    │
  │    ┌──────────┐            │                    │
  │    │ Critique │────────────┘                    │
  │    └────┬─────┴─────────────────────────────────┤
  │         │ Passed                                │ Round Limit
  │         ▼                                       │ Reached
  │    ┌──────────┐                                 │
  │    │  Judge   │                                 │
  │    └────┬─────┘                                 │
  │         │ Plan saved with hash                  │
  │         ▼                                       │
  │   ┌────────────┐                                │
  │   │   Await    │                                │
  │   │ Approval   │─── Rejected with feedback ──┐  │
  │   └─────┬──────┘                            │  │
  │         │ Approved                          │  │
  │         │                                Timeout│  │
  │         ▼                                   │  │
  │    ┌──────────┐                             │  │
  │    │ Execute  │                             │  │
  │    └────┬─────┘                             │  │
  │         │                                   │  │
  │         ▼                                   │  │
  │    ┌──────────┐  Check failed               │  │
  │    │  Verify  │─────────────────────────────┤  │
  │    └────┬─────┘                             │  │
  │         │ Healthy                           │  │
  │         │                                   ▼  ▼
  │         │                             ┌──────────┐
  │         │                             │ Escalate │
  │         │                             └────┬─────┘
  │         │                                  │
  │         ▼                                  │
  │    ┌──────────┐                            │
  │    │  Close   │<───────────────────────────┘
  │    └────┬─────┘
  │         │
  │         ▼
  │      ┌─────┐
  │      │ [*] │
  │      └─────┘

(Source)
3. Deterministic Guardrails Decision Flowchart
        ┌────────────────────────────────┐
        │     Tool Call from Agent       │
        └───────────────┬────────────────┘
                        │
                        ▼
         /──────────────────────────────\
        <  Server ID in allowlist?       >
         \──────────────────────────────/
             │                      │
            YES                     NO
             │                      │
             ▼                      ▼
  /───────────────────\    ┌─────────────────┐
 < Command ID in       >   │  DENY AND LOG   │
 < catalogue for this  >   └─────────────────┘
 < category?           >            ▲
  \───────────────────/             │
             │                      │
            YES                     NO
             │                      │
             ▼                      │
  /───────────────────\             │
 <  Parameters match   >────────────┤
 <  typed schema?      >            │
  \───────────────────/             │
             │                      │
            YES                     │
             │                      │
             ▼                      │
  /───────────────────\             │
 <  Command class is   >────────────┤
 <  read-only?         >            │
  \───────────────────/             │
             │                      │
            YES                     │
             │                      │
             ▼                      │
  /───────────────────\             │
 <  Rate and step      >────────────┘
 <  limits OK?         >
  \───────────────────/
             │
            YES
             │
             ▼
┌────────────────────────────────┐
│  Run with Read-Only Identity   │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│     Redact Secrets & PII       │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    Log & Return to Agent       │
└────────────────────────────────┘

(Source)
