# Chapter 1 — Introduction

`pp. 3–18` · *Building Generative AI Services with FastAPI* · Alireza Parandeh, O'Reilly 2025

A generative model is a dependency you pull, not something you train. The engineering is the
service around it: enrich the prompt from databases and external APIs before inference, check
the output before it reaches the user, and control access to both.

## What the chapter argues

Twelve of its sixteen pages argue that generative AI matters to application developers. The
engineering content sits on pp. 14–17: the service architecture, the framework comparison, the
adoption barriers, and the capstone scope.

The distinction that carries into the rest of the book is that traditional AI predicts and
classifies, while generative models produce content — text, code, images, audio, video, point
clouds, 3D meshes. Most models handle one modality; some are natively multimodal, like GPT-4o
over text, audio, and images. Sampling adds random noise so that outputs vary rather than
reproducing the training set, which is what makes a generative model probabilistic rather than a
fixed function. That property is also why chapter 11 cannot assert on output text.

Model families the chapter names (pp. 4–5):

| family | mechanism |
| ------ | --------- |
| Variational autoencoders | encode data into a low-dimensional latent space, decode back out when generating |
| Generative adversarial networks | a discriminator and a generator trained against each other; you keep the generator |
| Autoregressive models | predict the next value in a sequence from the previous ones |
| Normalizing flows | transform simple probability distributions into complex ones |
| Energy-based models | assign low energy to observed data and high energy to other configurations |
| Diffusion models | learn to add noise to data, then to remove it incrementally from a sampled point |
| Transformers | self-attention over sequences, parallelisable across the sequence; GPT is one |

Latent space (p. 5) is the compressed representation holding only what is needed to reconstruct
the input. Navigating it with a prompt yields content that was never in the training data,
because sampling interpolates between learned concepts rather than retrieving one.

## The service architecture the book is building toward

Figure 1-4, p. 14. The web server sits between the user and the model and does three jobs:

1. **Enrich.** Query databases and external services to add context to the prompt before it
   reaches the model.
2. **Control.** Decide who may call the model, and with what.
3. **Check.** Sanity-check the generated output before routing it back to the user.

Step 1 is where most of the value is. The chapter's analogy (pp. 9–10): searching for "ties"
forces a search engine to guess between clothing and knots, while "types of tie" does not. A
thin prompt gets a generic answer because generic is what best satisfies an under-specified
query. Detailed prompts get specific answers, and most of that detail should come from the
server, not from the user typing more.

The chapter also floats letting the model write an instruction for another system to execute —
a database query, an API call — and immediately points at chapter 9, because that is an attack
surface.

## Why FastAPI, in the chapter's own terms

Python is the constraint, not the preference: the deep-learning ecosystem lives there. `gin`
(Go) and `express` (Node.js) match FastAPI on performance but cannot hold the model in the same
process as the API.

Within Python the chapter weighs three (pp. 15–16):

- **Django** — mature, large community, MVC, Django REST Framework for APIs. Async support the
  chapter calls less mature, and enough overhead to be the wrong shape for a thin API.
- **Flask** — micro framework, leads on package downloads, extensible. Ships few defaults;
  schema validation is not among them.
- **FastAPI** — ships data validation, type safety, automatic OpenAPI documentation, and a web
  server. Lifecycle events give it a place to load model weights once, which chapter 3 uses.

Popularity figures on p. 15, dated at print: fastest-growing Python web framework by package
downloads, second most popular on GitHub, around 80,000 stars.

## What blocks adoption

The chapter's list, split by whether engineering can fix it (p. 16):

**Software engineering can fix these** — data privacy, cybersecurity, abuse and misuse of the
model, and integration with existing databases, web interfaces, and external APIs.

**These need prompt work or fine-tuning instead** — relevance, quality, coherence, and
consistency of the output.

Hallucination gets its own callout (p. 17): plausible-sounding output that is entirely made up.
It is why these models stay out of medical diagnosis, legal advice, and automated examinations.
The chapter is also candid that generative models recombine and rephrase rather than produce
genuinely unseen ideas, and that untuned output is generic and repetitive.

The book only claims the first half of that split. Chapters 7 through 9 are where it delivers.

## Numbers the chapter cites

Dated at print and more so now, but they are the concrete claims (pp. 10–11):

- Stack Overflow attributed a **~14% traffic decrease** to developers trialling GPT-4 after its
  release, with a **~60% decline** in questions asked and upvote activity against 2018.
- Stack Overflow's 2024 Developer Survey, 65,000 respondents: **72% favourable** toward AI
  tools, **43% trust their accuracy**.

## The capstone

What the service does by the end of the book (p. 17):

- three modalities — a language model for text and chat, an audio model for text-to-speech, and
  Stable Diffusion for images
- real-time responses as text, audio, or image
- RAG over uploaded documents through a vector database
- web scraping plus calls to internal databases, external systems, and APIs to gather context
- conversation history in a relational database
- token-based credentials and GitHub identity login
- authorization guards that restrict responses by permission
- guardrails against misuse and abuse

The UI is Streamlit and plain HTML on purpose, because the book is about the API. It names
React and Next.js as what you would reach for otherwise.

## For this repo

No code in this chapter; `examples/` starts at chapter 2.

The three jobs in Figure 1-4 are the outline of everything that follows, and worth keeping in
view while reading: enrichment is chapters 5 and 7, control is chapter 8, checking is chapter 9,
and chapter 10 makes all three cheaper.
