# Chapter 4 — Implementing Type-Safe AI Services

`pp. 117–148` · *Building Generative AI Services with FastAPI* · Alireza Parandeh, O'Reilly 2025

Type annotations do nothing at runtime, but FastAPI reads them to build the request contract,
and Pydantic turns them into checks that run on every request. One annotation therefore does
four jobs: editor autocomplete, `mypy`, runtime validation, and the OpenAPI schema.

## What the chapter argues

The case for typing is not correctness in the abstract. It is that a GenAI service talks to
things that change underneath it — a database schema, a provider's API, a model that starts
returning `None` — and without types you find out from a downstream stack trace that names the
wrong component. The chapter's own framing: what should have taken a minute takes half a day.

The progression is annotations → dataclasses → Pydantic, and each step exists because the
previous one runs out. Annotations describe; dataclasses group; only Pydantic validates.

## Annotations, `TypeAlias`, and `Annotated`

`Literal` plus a type alias is the cheapest way to close a set of values:

```python
# Example 4-2, trimmed
SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float]
price_table: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

def count_tokens(text: str | None) -> int:
    if text is None:
        logger.warning("Response is None. Assuming 0 tokens used")
        return 0
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))
```

Two habits from this example are worth keeping. Name types in `CamelCase` and mark them
`TypeAlias`, so they read as types rather than variable assignments. And **check at runtime
anyway** — the chapter raises `ValueError` for an unsupported model even though the `Literal`
already forbids it, because a type checker is advice and anyone can ignore it.

`Annotated` (Python 3.9+) does what a type alias does and carries metadata alongside:

```python
# Example 4-3
SupportedModels = Annotated[Literal["gpt-3.5-turbo", "gpt-4o"], "Supported text models"]
PriceTableType = Annotated[dict[SupportedModels, float], "Supported model pricing table"]
```

The metadata is invisible to type checkers but visible to Pydantic and to anything doing runtime
inspection, which is what makes `Annotated[str, Field(min_length=1)]` work later. It needs at
least two arguments: the type, then everything else. FastAPI's documentation prefers it over
plain aliases.

What FastAPI itself does with your annotations (p. 122):

- decides what is a path parameter, a query parameter, a body, a header, or a dependency
- converts incoming data to the annotated type
- validates data from requests, databases, and external services
- regenerates the OpenAPI specification and the `/docs` page

## Dataclasses, and where they stop

Dataclasses (Python 3.7+) group related parameters so a function signature stops bloating:

```python
# Example 4-4, trimmed
@dataclass
class Message:
    prompt: str
    response: str | None
    model: SupportedModels

def calculate_usage_costs(message: Message) -> MessageCostReport: ...
```

Four things they do not do, and all four matter for an API (p. 127):

| missing | what you notice |
| ------- | --------------- |
| Automatic parsing | an ISO datetime string stays a string |
| Field validation | nothing checks that the prompt is under 10,000 characters |
| Serialisation | JSON ↔ Python breaks on anything uncommon |
| Field filtering | no way to drop unset or `None` fields on export |

FastAPI accepts vanilla dataclasses and quietly converts them to Pydantic dataclasses to
validate and serialise. That is a migration path for an existing codebase, not a reason to start
with them: field constraints and computed fields stay out of reach.

## Pydantic models

A model is a class inheriting `BaseModel` with annotated attributes. Inheritance composes them,
which is how one request/response pair per modality stays readable:

```python
# Example 4-7, trimmed and with the timestamp default fixed
class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)]

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    ip: Annotated[str, IPvAnyAddress] | None
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]
    created_at: Annotated[datetime, Field(default_factory=lambda: datetime.now(timezone.utc))]

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]

class TextModelResponse(ModelResponse):
    tokens: Annotated[int, Field(ge=0)]
```

**`default_factory`, not a call.** `created_at: datetime = datetime.now()` evaluates once, when
the module is imported, so every response for the life of the process carries the same timestamp
— the moment the server started. A default must be a value or a callable that produces one, and
the callable form is `default_factory`. The `request_id` field above gets this right, which
makes the contrast easy to see.

Constrained types that arrive for free: `EmailStr`, `PositiveInt`, `UUID4`, `AnyHttpUrl`,
`HttpUrl`, `IPvAnyAddress`, and more. Some (`EmailStr`) need an extra package. Anything they do
not cover, `Field` covers: `ge`, `le`, `min_length`, `max_length`, `default`, `default_factory`,
`alias`.

When validation fails, FastAPI answers before your handler runs:

```json
{"detail": [{
  "type": "literal_error",
  "loc": ["body", "model"],
  "msg": "Input should be 'tinyllama' or 'gemma2b'",
  "input": "gpt-4o"
}]}
```

## Custom validators

Two shapes, and the difference is how many fields you can see.

**One field — `AfterValidator` in an `Annotated`, or `@field_validator`.** It runs after Pydantic
has parsed and validated the field, and can check or modify the value:

```python
# Example 4-9, trimmed
@validate_call
def is_square_image(value: ImageSize) -> ImageSize:
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)]
```

`@validate_call` on the validator itself makes it raise at runtime if someone calls it with the
wrong types directly, outside of Pydantic.

**More than one field — `@model_validator(mode="after")`.** A field validator cannot see its
siblings, so a rule like "tinysd may not exceed 2000 inference steps" belongs here:

```python
class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200

    @model_validator(mode="after")
    def check_inference_steps(self) -> "ImageModelRequest":
        if self.model == "tinysd" and self.num_inference_steps > 2000:
            raise ValueError("TinySD model cannot have more than 2000 inference steps")
        return self
```

The rule: `@field_validator` sees one value, `@model_validator` sees the whole instance. Reach
for the second only when the check genuinely spans fields.

## Computed fields

Derived values that live on the model rather than in the handler:

```python
# Example 4-10, with the decorator order Pydantic v2 expects
class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]

    @computed_field
    @property
    def tokens(self) -> int:
        return count_tokens(self.content)

    @computed_field
    @property
    def cost(self) -> float:
        return self.price * self.tokens
```

`@computed_field` goes above `@property`. Computed fields exist only on export — `.model_dump()`
or the serialisation FastAPI does when returning the model — so they cost nothing until someone
asks for them, and they appear in the API response with no extra work in the controller.

## Export and field filtering

```python
>>> response = TextModelResponse(content="FastAPI Generative AI Service", ip=None)
>>> response.model_dump(exclude_none=True)
{'content': 'FastAPI Generative AI Service', 'cost': 0.06, 'price': 0.01,
 'request_id': 'a3f18d85dcb442baa887a505ae8d2cd7', 'tokens': 6, 'created_at': ...}

>>> response.model_dump_json(exclude_unset=True)
'{"ip":null,"content":"FastAPI Generative AI Service","tokens":6,"cost":0.06}'
```

Three filters, and the distinction is worth getting right the first time:

| flag | keeps |
| ---- | ----- |
| `exclude_none=True` | everything that is not `None` |
| `exclude_unset=True` | only fields explicitly passed at construction |
| `exclude_defaults=True` | only fields differing from their default |

`exclude_unset` is the one that matters for filter and query models: whatever the client did not
send stays out of the query.

## Settings from the environment

`pydantic-settings` is a separate install. `BaseSettings` fills each field from the constructor,
then from environment variables, then from the default — and validates the result like any other
model, so a malformed `DATABASE_URL` fails at startup instead of at first query.

```python
# Example 4-12
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[PostgresDsn, Field(
        alias="DATABASE_URL", default="postgres://user:pass@localhost:5432/database")]
    cors_whitelist_domains: Annotated[set[HttpUrl], Field(
        alias="CORS_WHITELIST", default=["http://localhost:3000"])]

settings = AppSettings()
```

`snake_case` field names map to `UPPER_CASE` variables automatically; `alias` overrides that when
the variable is already named something else. `AppSettings(_env_file="test.env")` swaps the file,
which is how the test suite gets its own configuration.

## The typed handler

Everything above collapses into a controller that declares its contract and does nothing else:

```python
# Example 4-14
@app.post("/generate/text")
def serve_text_to_text_controller(
    request: Request, body: TextModelRequest = Body(...)
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]:
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

The handler never touches `tokens` or `cost` — the computed fields produce them during
serialisation. `request.client.host` is where the client IP comes from, and `Request` is
injected by declaring it. The `GET` with query parameters from chapter 3 becomes a `POST` with a
body, because a validated schema needs somewhere to live.

Response, for reference:

```json
{"request_id": "7541204d5c684f429fe43ccf360f33dc", "ip": "127.0.0.1",
 "content": "...", "created_at": "2024-03-07T16:06:57.492039",
 "price": 0.01, "tokens": 25, "cost": 0.25}
```

## For this repo

Four files in [`examples/`](examples/), following the chapter:

| file | shows |
| ---- | ----- |
| `01_type_error.py` | the annotation `mypy` catches and the interpreter does not |
| `02_using_type_annotations.py` | `Literal`, `TypeAlias`, the price table, `count_tokens` |
| `03_dataclasses.py` | the same logic with `Message` and `MessageCostReport` |
| `04_pydantic.py` | `BaseModel`, compound models, constrained fields |

Two things to fix in them before moving on:

- **`04_pydantic.py`: `created_at: datetime = datetime.now()`.** Evaluated once at import, so
  every response carries the moment the process started. Use
  `Annotated[datetime, Field(default_factory=lambda: datetime.now(timezone.utc))]`. The same
  line recurs in chapter 7's SQLAlchemy columns, so it is worth fixing the habit here.
- **`03_dataclasses.py` imports `count_tokens` from `utils`**, but the function lives in
  `02_using_type_annotations.py` and there is no `utils.py`. Either pull the shared helpers into
  a real `utils.py` beside them, or define it locally.

Still to build:

1. `schemas.py` — the compound `ModelRequest` / `ModelResponse` hierarchy with constrained
   fields, computed `tokens` and `cost`, and `default_factory` on every generated default.
2. `settings.py` — `AppSettings(BaseSettings)` replacing the `os.environ[...]` reads in
   `chapter_02/examples/01_sample_app.py`, so a missing or malformed variable fails at startup
   with a Pydantic error rather than a `KeyError` mid-request.
3. `utils.py` — `count_tokens` via `tiktoken`, and a cost calculation with the price table as a
   `Literal`-keyed dict.
4. The chapter-3 controllers converted to `POST` with typed bodies and typed responses.

`mypy` is already `strict = true` in `pyproject.toml` with the `pydantic.mypy` plugin and
`init_typed`, so the plugin will check model constructor calls too — which is most of the value
of this chapter, enforced automatically.

One promise to not wait for: p. 120 says Prisma and API client generators, for auto-generating
types from a database schema or an external API, are covered "later in the book". Neither
arrives. Chapter 7 builds SQLAlchemy and Alembic instead.

---

*Study notes from* **Building Generative AI Services with FastAPI** *by Alireza Parandeh
(O'Reilly, 2025) — summarised for personal reference, with short cited excerpts. All ideas and
examples are the author's; see [Attribution](../README.md#attribution). Buy the book:
<https://buildinggenai.com>*
