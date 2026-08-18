# Chapter 2 — Getting Started with FastAPI

`pp. 19–52` · *Building Generative AI Services with FastAPI* · Alireza Parandeh, O'Reilly 2025

FastAPI is a wrapper over Starlette with Pydantic doing validation and serialisation, so an
endpoint is a decorated function and nothing else. The chapter's most useful seven pages are the
ones listing what FastAPI cannot do for AI workloads, which is what makes chapters 3, 5, and 12
necessary.

## What the chapter argues

An ASGI framework processes requests on an event loop rather than one-per-worker, so the same
process can hold a model in memory and still serve concurrent requests. That combination — the
Python ML ecosystem in-process, plus concurrency — is the case for FastAPI over `gin` or
`express`, and the case for FastAPI over Flask (WSGI, synchronous, no native WebSocket).

The chapter then spends its second half undermining its own recommendation: seven limitations,
of which three bite as soon as the model gets large. That honesty is what makes it worth reading
twice.

Everything else is scaffolding for the rest of the book: dependency injection (chapters 5, 7, 8
all lean on it), lifespan events (chapter 3), and the layered design that keeps model code out
of route handlers.

## The whole server in fifteen lines

```python
# Example 2-1, trimmed — see examples/01_sample_app.py for the version with the key in .env
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
openai_client = OpenAI(api_key="your_api_key")

@app.get("/")
def root_controller():
    return {"status": "healthy"}

@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return {"statement": response.choices[0].message.content}
```

`fastapi dev` finds the `app` object, starts `uvicorn` on `127.0.0.1:8000`, and runs a file
watcher that reloads on save. The decorator turns the function into an HTTP endpoint; the return
value is serialised to JSON, because HTTP moves only text or binary.

Two things come free and are worth knowing about before you need them:

- `/docs` — a Swagger UI generated from an `openapi.json` the framework also generates. Faster
  than writing a test while you are still designing the route signature; not a replacement for
  one afterwards. **Turn it off in production** unless the API is public, or it advertises your
  unsecured endpoints.
- **Validation and serialisation via Pydantic**, which catches at runtime what `mypy` cannot see
  at all — an eight-character password rule, a real email address, a UUID.

```python
# Example 2-2, corrected for Pydantic v2 — see examples/02_pydantic_valid.py
from pydantic import BaseModel, field_validator

class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("password")
    def validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isdigit() for c in value):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in value):
            raise ValueError("Password must contain at least one uppercase letter")
        return value
```

## Where `def` and `async def` handlers actually run

This is the single sentence to carry forward from the chapter (p. 25):

- `async def` handler → runs on the **main thread, on the event loop**.
- `def` handler → runs on a **thread from the internal pool**, so it cannot block the loop.

The chapter adds that threads carry overhead, so a service that is all-synchronous still hits a
scaling wall. Chapter 5 is where this turns into a rule you can apply; for now the point is that
the framework treats the two keywords as different execution models, not as style.

## Dependency injection

Inversion of control: a handler declares what it needs, and the framework builds it. Two shapes
matter.

A plain function for shared parameters — one definition, every route that wants pagination:

```python
# Example 2-4 — examples/04_dependency_injection.py
def paginate(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/messages")
def list_messages_controller(pagination: dict = Depends(paginate)):
    ...
```

A generator for resources that need cleanup — the code after `yield` runs once the response is
sent:

```python
# Example 2-5 — examples/05_dependency_db.py
def get_db():
    db = ...              # create a database session
    try:
        yield db
    finally:
        db.close()        # runs after the request, so connections do not leak

@app.get("/users/{email}/messages")
def get_user_messages(email, db=Depends(get_db)):
    user = db.query(...)      # same session
    messages = db.query(...)  # same session
    return messages
```

**Dependencies are cached per request.** A dependency function runs once per request no matter
how many places ask for it, and runs again on the next request. That is what makes it safe to
put "fetch the current user" behind a dependency and depend on it from four places — chapter 8
builds exactly that. Dependencies can also depend on other dependencies, forming a graph
(Figure 2-2).

## Lifespan events

Startup and shutdown hooks for anything shared between requests: database connection pools, and
model weights. Load the model once at startup, keep it for the life of the process, unload it on
shutdown after in-flight requests drain. Chapter 3 builds this; the note here is that FastAPI's
lifespan replaced the older `startup`/`shutdown` events, so any example using those is out of
date.

## Project structure: flat, then nested, then modular

The chapter's advice is to move between three structures as complexity arrives rather than
picking the final one up front. Layouts are in [`examples/06_project_structures.md`](examples/06_project_structures.md).

| stage | grouping | when |
| ----- | -------- | ---- |
| **Flat** | everything at the root of `app/` — `models.py`, `routers.py`, `services.py` | first version, or a genuine microservice. No coupling to think about because there is barely any code |
| **Nested** | packages by *kind* — `models/users.py`, `routers/users.py` | what the official FastAPI docs recommend for larger projects. Pitfall named in the book: ambiguous coupling, where one change forces edits everywhere ("shotgun updates") |
| **Modular** | packages by *domain* — `modules/auth/{routers,models,services,guards}.py` | full backend. Popularised by the Netflix Dispatch project. Adding a feature means adding a package |

The test the chapter offers: if you cannot justify the file layout to another developer, it is
wrong. If you spend hours looking for a piece of code, it is wrong.

## The onion layers

Separation of concerns with a dependency direction: domain models sit at the centre, every outer
layer depends inward, and nothing inner knows about anything outer. FastAPI's dependency system
is the mechanism that keeps the direction honest, because a high-level module declares what it
needs instead of importing the implementation.

Layers, outermost first (pp. 40–42):

| component | responsibility |
| --------- | -------------- |
| **API routers** | group controllers so common logic applies to several at once (`APIRouter`) |
| **Controllers** (route handlers) | take the request, orchestrate services and providers, return the response |
| **Services** | business logic composed of several internal operations |
| **Providers** | the same idea for *external* systems — email servers, payment gateways, other services |
| **Repositories** | data access and mutation against a store, via ORM or raw SQL. CRUD is the usual interface |
| **Schemas / models** | type safety, structure, and validation on data crossing every layer |

Six more components cut across every layer:

| component | responsibility |
| --------- | -------------- |
| **Middleware** | runs before and after the controller — headers, logging, CORS |
| **Dependencies** | injectable reusable functions, cached per request |
| **Pipes** | data transformers used anywhere — aggregators, cleaners, parsers |
| **Mappers** | translate between schemas across layers, e.g. `UserRequest` → `UserInDB` |
| **Exception filters** | one consistent way to turn errors into responses |
| **Guards** | protect controllers; authentication and authorization as dependencies or middleware |

The modular project layout is this list turned into directories, which is why the two sections
are next to each other in the chapter.

## FastAPI against Django and Flask

The axis is opinionated versus not. Django and Nest.js decide for you; FastAPI, Flask, and
Express hand you the freedom and the decision fatigue that comes with it. Working with a
database in FastAPI means choosing and integrating three packages — a driver, a migration tool,
and an ORM — where Django ships one that already works. Chapter 7 pays that bill.

- **Django** — ORM with migrations, admin panel, credentials-based auth, web security defaults,
  MVC. Async since 4.2. The right answer for a monolith serving its own frontend; heavy for a
  lean API.
- **Flask** — released 2010, WSGI, so each request is processed synchronously and a call to an
  external API blocks a whole worker process. No WebSocket support without extensions, because
  WSGI has none. No validation, no auto-documentation, no dependency injection.
- **Quart** — Flask on ASGI, named as a genuine contender, ruled out on community size.

## Seven limitations for AI workloads

The chapter's own list (pp. 46–49), and the reason the rest of the book exists:

1. **No shared model memory across processes.** Scaling to N web workers means N full copies of
   the weights in the container's memory.
2. **A ceiling on threads.** FastAPI uses AnyIO, which creates **up to 40 threads** by default
   in a dynamic internal pool. Every synchronous handler occupies one while it runs.
3. **The GIL.** Threads do not run Python bytecode in parallel, so a CPU-bound inference call
   blocks the others. Expensive compute needs multiprocessing or a process pool, not threads.
4. **No micro-batching.** Deep learning frameworks vectorise across a batch; FastAPI has no way
   to group concurrent inference requests into one, so each blocks the next.
5. **No CPU/GPU split.** Preparation and post-processing belong on the CPU, inference on the
   GPU. FastAPI cannot separate them, so the CPU stays blocked while the GPU works.
6. **Dependency conflicts.** Model runtimes couple tightly to native libraries and specific
   hardware, which ordinary web deployments never have to think about.
7. **It predates generative AI.** A general-purpose web framework with model-serving support
   added late.

The escape hatch named here and built later: **serve heavy models outside FastAPI** and keep
FastAPI for auth, caching, and business logic. **BentoML** is the book's candidate — also built
on Starlette, also FastAPI-shaped, but with Runners that scale web requests separately from
inference, model versioning, and auto-generated Dockerfiles that handle the CUDA install.

## Tooling the chapter recommends

Environment: `requirements.txt` with pip for simple projects, `uv` or Conda for pip-driven
workflows, Poetry for complex ones. The book's examples are tested against **Python 3.11**.

Everything else collapses into two tools worth actually installing: **Ruff** (Rust-based,
replaces isort, Black, Flake8, and much of Bandit) and **mypy**. The chapter also lists Autoflake,
Flake8, isort, Black, Loguru, Bandit, Safety, and Pylance, and recommends running the checks from
a pre-commit hook rather than trusting yourself to remember.

## For this repo

This project already goes further than the chapter: `uv` instead of pip, Python 3.12 instead of
3.11, Ruff and mypy in `pyproject.toml` with `ASYNC` and `S` rules enabled, and `pre-commit`.
Ruff's `extend-immutable-calls` list is there because FastAPI puts callables in argument defaults
on purpose, which trips `B008` otherwise.

Five runnable examples in [`examples/`](examples/):

| file | shows |
| ---- | ----- |
| `01_sample_app.py` | the fifteen-line server, key and model read from `.env` |
| `02_pydantic_valid.py` | field validation, on Pydantic v2's `@field_validator` |
| `03_auto_docs.py` | `/` redirecting to `/docs` with a 303 |
| `04_dependency_injection.py` | a shared-parameter dependency |
| `05_dependency_db.py` | a `yield` dependency that closes its session |

Structure decision for the capstone: start flat, reorganise when it hurts. The complexity is not
knowable yet, and two restructuring passes later cost less than guessing the final shape now.
