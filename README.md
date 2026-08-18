# GenAI-with-Fast-API

Working through *Building Generative AI Services with FastAPI* (Alireza Parandeh, O'Reilly
2025). One directory per chapter: `README.md` holds the notes, `examples/` holds the runnable
code.

## Attribution

**These are personal study notes. The book itself is:**

> **Building Generative AI Services with FastAPI**
> Alireza Parandeh · O'Reilly Media, 2025
> Companion site: <https://buildinggenai.com>
> Author's code: <https://github.com/Ali-Parandeh/building-generative-ai-services>

Every idea, argument, chapter structure, and worked example in `chapter_*/README.md` originates
with the author. What I contributed is the summarising, the condensing, and the occasional note
where an example needed correcting to run. Code snippets are short excerpts, trimmed to the
point being made and cited by the book's own example numbers, kept here for study under fair
use. Copyright remains with the author and O'Reilly Media.

## Chapters

| ch | title | notes | code |
| -- | ----- | ----- | ---- |
| 01 | Introduction | [notes](chapter_01/README.md) | — |
| 02 | Getting Started with FastAPI | [notes](chapter_02/README.md) | [examples](chapter_02/examples) |
| 03 | AI Integration and Model Serving | [notes](chapter_03/README.md) | [examples](chapter_03/examples) |
| 04 | Implementing Type-Safe AI Services | [notes](chapter_04/README.md) | [examples](chapter_04/examples) |
| 05 | Achieving Concurrency in AI Workloads | [notes](chapter_05/README.md) | — |
| 06 | Real-Time Communication with Generative Models | [notes](chapter_06/README.md) | — |
| 07 | Integrating Databases into AI Services | [notes](chapter_07/README.md) | — |
| 08 | Authentication and Authorization | [notes](chapter_08/README.md) | — |
| 09 | Securing AI Services | [notes](chapter_09/README.md) | — |
| 10 | Optimizing AI Services | [notes](chapter_10/README.md) | — |
| 11 | Testing AI Services | [notes](chapter_11/README.md) | — |
| 12 | Deployment of AI Services | [notes](chapter_12/README.md) | — |

Part I is chapters 1–4 (the framework), Part II is 5–7 (external systems), Part III is 8–12
(production).

## Reading order, if you are not going straight through

Chapter 5 is the spine: it explains why inference blocks, why the ceiling is GPU memory, and why
the model eventually leaves the FastAPI process. Chapters 3, 10, and 12 are the same argument at
different altitudes. Chapter 11's behavioural testing section is the material hardest to find
elsewhere.


```
uv sync
uv run ruff check --fix .
uv run ruff format .
uv run mypy .
```
