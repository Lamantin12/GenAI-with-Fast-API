# Chapter 3 — AI Integration and Model Serving

`pp. 53–116` · *Building Generative AI Services with FastAPI* · Alireza Parandeh, O'Reilly 2025

Five modalities behind five endpoints, then the question that decides everything about the
service: **when do the weights get loaded?** Per request, once at startup, or never — because
another process holds them.

## What the chapter argues

Serving a generative model is two decisions. The first is the response type: text goes back as
JSON, everything else is binary, and audio and video are large enough to stream rather than
buffer. The second is the model-serving strategy, and the chapter lays out three points on a
ladder — load per request, preload in the lifespan, serve externally — which is the escalation
the rest of the book follows.

The model-architecture sections are background rather than instructions. They matter in two
places: the **context window** determines what chapter 7's conversation history has to fit
inside, and the fact that transformers generate **one token at a time** is what makes chapter 6's
streaming possible at all.

Everything here uses tiny models on purpose (TinyLlama, bark-small, tiny-sd) so the examples run
on a CPU. The hardware boxes are the honest part.

## How each model works, compressed

**Language models.** RNNs carry a state vector from token to token, so early tokens fade by the
end of a long sequence, and training cannot parallelise because it is inherently sequential.
Transformers drop the hidden state for **self-attention**, mapping pairwise relationships between
every token regardless of distance, which also makes training parallelisable on GPUs. Attention
heads each compute their own attention map; multiple heads per layer, multiple layers per model.

The pipeline from text to prediction:

1. **Tokenize** — slice text into words, syllables, symbols, punctuation; map each to an integer.
2. **Embed** — convert each token id into a dense float vector. Distance in that space is
   meaning; cosine similarity between two embeddings measures it.
3. **Positionally encode** — since attention sees the whole sequence at once, add a positional
   embedding to each token embedding so word order survives.
4. **Predict autoregressively** — one token at a time, each conditioned on everything before,
   until `<stop>` or `<eos>`.

**Context window** is the token limit the model can hold while predicting the next one. Overflow
discards least-recently-used tokens, so the model forgets the start of a long conversation.
Short windows lose information; long ones cost memory, latency, and money.

Three transformer shapes, and picking the wrong one is a real mistake:

| shape | good at |
| ----- | ------- |
| Encoder–decoder | sequence-to-sequence — translation, summarisation, question answering |
| Encoder-only | understanding input — sentiment, entity extraction, classification |
| Decoder-only | predicting the next token — text generation and chat |

**Audio (Bark, Suno AI).** Four chained models: a causal autoregressive transformer turns
tokenized text into semantic tokens; a second causal transformer turns those into coarse audio
features; a non-causal autoencoder transformer fills in the fine features (non-causal because the
whole sequence already exists); the Encodec codec decodes the audio array. The output is
amplitude values over time plus a sample rate, which `soundfile` turns into a WAV.

**Vision (Stable Diffusion).** Training encodes images into a latent space and learns to remove
noise added to them. Generation runs the reverse: perturb a point in the latent space, denoise it
over several iterations, decode. Text prompts steer it because image descriptions were encoded
alongside the images, mapping regions of the latent space to language. `num_inference_steps`
trades compute for quality directly.

**Video.** Image-to-video via Stable Video Diffusion, which needs a real GPU. The chapter's aside
on Sora is the interesting part: a 3D U-Net where the third dimension is time, predicting the
next **visual patch** the way a language model predicts the next token, which gives it 3D
consistency, object permanence across occlusion, and world interaction as emergent properties.

**3D (Shap-E, OpenAI).** Predicting vertices as a sequence needs thousands of them for a smooth
surface and takes far too long. Shap-E instead trains an encoder that produces an **implicit
function** defining surfaces continuously. The decoder renders with NeRF — map a 3D coordinate
and viewing direction to density and RGB, integrate along each camera ray — and **signed distance
functions** turn the scene into a mesh (negative inside, zero on the surface, positive outside).

## Serving each modality

The pattern is identical across models: a `load_*` function and a `generate_*` function in
`models.py`, a buffer helper in `utils.py`, a controller in `main.py`. What changes is the
response class.

| modality | returns | why |
| -------- | ------- | --- |
| text | plain `str` → JSON | small enough that none of this matters |
| image | `Response(content=..., media_type="image/png")` | binary, but small |
| audio | `StreamingResponse(BytesIO, media_type="audio/wav")` | large; the client can play while it arrives |
| video | `StreamingResponse(BytesIO, media_type="video/mp4")` | larger still |
| 3D | `StreamingResponse` plus `Content-Disposition: attachment` | it is a file download |

Two annotations matter and are easy to skip:

```python
@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},  # documents the media type
    response_class=Response,   # stops FastAPI also advertising application/json
)
```

Without `response_class`, the OpenAPI schema claims the endpoint can return JSON, which it
cannot.

Loading a model and generating, in the shape every modality uses:

```python
# Example 3-1, trimmed — models.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_model():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device=device,
    )

def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    predictions = pipe(
        prompt, temperature=temperature, max_new_tokens=256,
        do_sample=True, top_k=50, top_p=0.95,
    )
    return predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
```

Inference parameters, which are the same five everywhere:

- `max_new_tokens` — cap on generated length.
- `do_sample` — `True` picks randomly among candidates, `False` always takes the most likely
  token. Set it `False` and the model becomes deterministic, which chapter 11 needs.
- `temperature` — how flat the sampling distribution is. Lower is more predictable.
- `top_k=50` — sample only from the 50 most likely tokens.
- `top_p=0.95` — nucleus sampling: keep adding candidates until they cover 95% of the
  probability mass, then sample from those.

Precision is not a free choice. TinyLlama runs in `bfloat16` to halve memory. Stable Diffusion
uses `float32` because `float16` loses too much precision for the model and CPU support for it is
thin. The video model demands `float16` with `variant="fp16"`, which is why it needs a GPU.

Streaming audio out of memory:

```python
# Example 3-5, trimmed — utils.py + main.py
def audio_array_to_buffer(audio_array: np.ndarray, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav")
    buffer.seek(0)              # rewind, or the client gets zero bytes
    return buffer

@app.get("/generate/audio", response_class=StreamingResponse)
def serve_text_to_audio_model_controller(prompt: str, preset: VoicePresets = "v2/en_speaker_1"):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")
```

Buffer in memory or write to a file first and stream from disk — the chapter frames it as
trading latency for memory, and defaults to memory.

The Streamlit client is 20 lines and swaps one call per modality: `st.markdown`, `st.audio`,
`st.image`. `st.session_state.messages` holds the history, `st.chat_input` blocks until the user
submits.

## The three serving strategies

This is the chapter's most reusable section (pp. 102–110).

**1. Model-agnostic — load on every request.** Load, generate, unload, repeat. Memory is free
between requests, so you can swap between models larger than RAM. Requests queue FIFO and each
one pays the full load cost. For prototyping on a weak machine only; the chapter says never in
production, and it is right.

**2. Compute-efficient — preload in the lifespan.** One load per process, reused by every
request. Costs RAM (or VRAM) for the life of the process, saves minutes per request.

```python
# Example 3-16 — examples/model_preloading_with_lifespan.py
models = {}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    models["text2image"] = load_image_model()   # before yield: startup
    yield                                       # requests are served here
    models.clear()                              # after yield: shutdown

app = FastAPI(lifespan=lifespan)

@app.get("/generate/image", response_class=Response)
def serve_text_to_image_model_controller(prompt: str):
    output = generate_image(models["text2image"], prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")
```

The dictionary at module scope is the whole trick: the lifespan fills it before the first request
and empties it after the last. `asynccontextmanager` replaced the older `@app.on_event("startup")`
and `@app.on_event("shutdown")` handlers in FastAPI 0.93.0, so anything using those is out of
date — see `examples/model_preloading_with_lifespan_LEGACY.py` for what to recognise.

You can preload several models this way, and the chapter immediately warns not to: consumer GPUs
top out at 24 GB VRAM and a single model can want 18 GB. Separate instances, separate GPUs.

**3. Lean — serve the model somewhere else.** FastAPI keeps coordination, auth, monitoring,
content filtering, and prompt enrichment; something else holds the weights. Three flavours:

- **BentoML** — the book's preferred answer. Runs different requests in different worker
  processes, so CPU-bound inference parallelises without you touching `multiprocessing`, and it
  batches inferences so several users' requests become one model call. Both of those are
  limitations 3 and 4 from chapter 2, fixed.
- **Cloud model endpoints** — Azure ML Studio's PromptFlow and similar. The chapter is blunt
  about the learning curve.
- **Model providers** — OpenAI and friends, where FastAPI becomes a wrapper. LangChain if you
  want to swap providers without rewriting.

```python
# Examples 3-18 and 3-19, trimmed — examples/bento.py
@bentoml.service(resources={"cpu": "4"}, traffic={"timeout": 120}, http={"port": 5000})
class Generate:
    def __init__(self) -> None:
        self.pipe = load_image_model()

    @bentoml.api(route="/generate/image")
    def generate(self, prompt: str) -> str:
        return self.pipe(prompt, num_inference_steps=10).images[0]

# FastAPI becomes the client
@app.get("/generate/bentoml/image", response_class=Response)
async def serve_bentoml_text_to_image_controller(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/generate", json={"prompt": prompt})
    return Response(content=response.content, media_type="image/png")
```

The moment the model moves out of process, the call becomes network I/O, so the handler can be
`async def` with `httpx.AsyncClient` and stops blocking anything. That is not incidental — it is
half the reason the strategy works.

Anything leaving your process leaves your privacy boundary too. Managed offerings like Azure
OpenAI exist for that reason; self-hosting trades the privacy back for operational work.

## Middleware for monitoring

Middleware runs before the controller sees the request and after it produces the response, which
makes it the right place for anything that applies to every endpoint: logging, rate limiting,
content filtering, CORS.

```python
# Example 3-22, trimmed — examples/logging_middleware.py
@app.middleware("http")
async def monitor_service(
    req: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = uuid4().hex
    start_time = time.perf_counter()
    response: Response = await call_next(req)          # the whole rest of the app happens here
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id
    ...                                                # append a row to usage.csv
    return response
```

The signature is fixed: a `Request` and a `call_next` callback, and `await call_next(req)` is the
line where every route handler runs. Generate the request id **before** that call so a failure
inside it is still traceable.

Logging to a CSV on disk survives nothing — restart the container without a mounted volume and
the file is gone. Chapter 7 moves it to a database. Logging prompt and response *bodies* is the
efficient thing to do from middleware and the risky thing to do at all, since users put sensitive
data in prompts.

## Numbers worth keeping

| | |
| --- | --- |
| TinyLlama | 1.1B parameters, pretrained on 3T tokens, ~3 GB disk and RAM, ~1 minute per response on CPU |
| tiny-sd | ~5 GB disk and RAM on CPU; `pip install accelerate` lowers the peak |
| SDXL | 16 GB CPU RAM **and** 16 GB GPU VRAM — loaded to CPU from disk, then moved to GPU |
| Largest open LLM at print | Snowflake Arctic, 480B, wants an 8×H100 instance (80 GB VRAM per card); Llama 3.1 405B similar |
| Best consumer GPU at print | RTX 4090, 24 GB VRAM — not enough above ~30B unquantized; a quantized 70B Llama wants 64 GB |
| What most organisations actually run | models up to ~3B, or a provider API |
| Context windows | `gpt-4o-mini` ~128K tokens ≈ 300+ pages; largest at March 2025 was Magic.Dev LTM-2-mini at 100M tokens ≈ 10M lines of code |
| Video encoding | h264 at 30 fps, `yuv444p`, CRF 17 (near-lossless), input resized to 1024×576, 25 frames, `decode_chunk_size=8` |

Fine-tuning is out of scope but LoRA gets a box: train a small number of new parameters per
layer, freeze the rest, and the GPU memory needed for fine-tuning drops enough to be feasible.

Stable Diffusion's limitations at print, worth setting expectations against: incomplete
coherency on complex prompts, fixed output sizes (512×512, 1024×1024), no real compositional
control, visible AI artefacts, and illegible text.

## For this repo

[`examples/`](examples/) has the full service:

| file | contents |
| ---- | -------- |
| `models.py` | `load_*` / `generate_*` for text, audio, image, video, 3D |
| `schemas.py` | `VoicePresets` as a `Literal` — the whole schema layer, for now |
| `utils.py` | `audio_array_to_buffer`, `img_to_bytes`, `export_to_video_buffer`, `mesh_to_obj_buffer` |
| `main.py` | the five serving endpoints |
| `model_preloading_with_lifespan.py` | strategy 2, and `_LEGACY.py` for the pre-0.93 form |
| `bento.py` | strategy 3, the BentoML service |
| `logging_middleware.py` | the usage-logging middleware |
| `client_text.py`, `client_audio.py`, `client_image.py` | the Streamlit clients |

Decision for the capstone: **preload in the lifespan**. One load per process without running a
second service, which is the right middle for models small enough to co-reside. Revisit when
inference starts blocking — chapter 5 supplies the reason it will, and `bento.py` is already the
escape hatch.

Note that `main.py` still calls `load_*` inside each controller. That is strategy 1, kept because
the chapter builds it that way; the lifespan version lives in its own file. Merging them is the
first thing to do before chapter 5.
