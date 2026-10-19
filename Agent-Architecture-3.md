```text
+---------------------------------------------------------------------------------------------------------------+
|                                                Hermes Harness                                                 |
|                                         (local / Docker / SSH / VPS)                                          |
|                                                                                                               |
|  +---------------------+                                +-------------------------------------------------+   |
|  |  Gateway Interface  |                                |                      Loop                       |   |
|  |  [Telegram] [WhatsApp]                               |  +-------------------------------------------+  |   |
|  |  [Slack] [TUI/Matrix]                                |  |            Hermes Agentic Tools           |  |   |
|  +----------+----------                                 |  |  - Terminal, Browser, Cron, MCP, etc.     |  |   |
|             |                                           |  +---------------------+---------------------+  |   |
|             |                                           |                        |     Tool Calls /           |   |
|             v                                           |                        v     Responses              |   |
|  +-----------------------------------------------------------------------------------+                        |   |
|  |                               Ephemeral Agent Run                                 |                        |   |
|  |                                                                                   |                        |   |
|  |     +--------------+      +------------------+                                    |                        |   |
|  |     | User Prompt  |----->|  Working Memory  |<----+                              |                        |   |
|  |     +--------------+      +--------+---------+     |                              |                        |   |
|  |                                    |               |                              |                        |   |
|  |     +--------------+               v               |                              |                        |   |
|  |     | System Prompt|<------- [LLM - Q&A Agent] ----+                              |                        |   |
|  |     +--------------+               |                                              |                        |   |
|  |                                    v                                              |                        |   |
|  |                                 [Reply] --------> Save History -----------------+ |                        |   |
|  +------------------------------------+--------------------------------------------+ |                        |   |
|                                       |                                              |                        |   |
|             +-------------------------+-------------------------+                    |                        |   |
|             |                         |                         |                    |                        |   |
|             v                         v                         v                    v                        |   |
|      +--------------+          +--------------+          +--------------+     +-------------------+           |   |
|      |  Procedural  |          |   Semantic   |          |   Episodic   |     | Sub-Agents &      |           |   |
|      |    Memory    |          |    Memory    |          |    Memory    |     | Delegate Tasks    |           |   |
|      |  (SKILL.md)  |          | (MEMORY.md)  |          |  (state.db)  |     | (Claude Code, etc)|           |   |
|      +--------------+          +--------------+          +--------------+     +-------------------+           |   |
+---------------------------------------------------------------------------------------------------------------+
```

---

### Questions & How Hermes Addresses Them

* **Q1: How do I find my OS version and recent shell history?**
* **Hermes Solution:** Uses native agentic tools for the terminal and filesystem to execute commands and inspect logs.


* **Q2: How do I browse a webpage and extract metrics/titles?**
* **Hermes Solution:** Uses built-in browser and web tools to fetch and parse web content.


* **Q3: How does Hermes save and recall user preferences?**
* **Hermes Solution:** Stores persistent facts via semantic memory files (`MEMORY.md`) and updates contextual files.


* **Q4: How do I search past session history?**
* **Hermes Solution:** Queries SQLite-backed session storage (`state.db`) with full-text search (FTS5).


* **Q5: How does Hermes handle agent identity and system prompts?**
* **Hermes Solution:** Loads personality and behavior rules dynamically from `SOUL.md`.


* **Q6: How does Hermes learn new procedures or tasks?**
* **Hermes Solution:** Self-builds and updates procedural memory files (`SKILL.md`) using markdown-based instructions.


* **Q7: How do I delegate tasks to multiple background workers?**
* **Hermes Solution:** Spawns isolated sub-agents in parallel using the `delegate_task` tool.


* **Q8: How do I schedule recurring background tasks?**
* **Hermes Solution:** Uses cron jobs (`crontab`) to run scheduled agent triggers automatically.


* **Q9: How does Hermes connect across different chat platforms?**
* **Hermes Solution:** Uses modular messaging gateways (Telegram, WhatsApp, Slack, etc.) mapped to the same core agent memory and logic.
