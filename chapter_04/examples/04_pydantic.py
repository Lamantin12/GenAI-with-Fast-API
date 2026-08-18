# --------------------------------
# Basemodel
# --------------------------------

from typing import Literal
from pydantic import BaseModel


class TextModelRequest(BaseModel):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0


# --------------------------------
# Compound models
# --------------------------------
    

# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel


class ModelRequest(BaseModel):
    prompt: str


class ModelResponse(BaseModel):
    request_id: str
    ip: str | None
    content: str | bytes
    created_at: datetime = datetime.now()


class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: float = 0.0


class TextModelResponse(ModelResponse):
    tokens: int


ImageSize = Annotated[tuple[int, int], "Width and height of an image in pixels"]


class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: int = 200


class ImageModelResponse(ModelResponse):
    size: ImageSize
    url: str


# --------------------------------
# Constrained fields
# --------------------------------
    
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt


class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)]


class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    # no defaults set for ip field
    # raise ValidationError if a valid IP address or None is not provided.
    ip: Annotated[str, IPvAnyAddress] | None
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]
    created_at: datetime = datetime.now()


class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]


class TextModelResponse(ModelResponse):
    tokens: Annotated[int, Field(ge=0)]


ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]


class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200


class ImageModelResponse(ModelResponse):
    size: ImageSize
    url: Annotated[str, HttpUrl] | None = None

# $ curl -X 'POST' \
#   'http://127.0.0.1:8000/validation/failure' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "prompt": "string",
#   "model": "gpt-4o",
#   "temperature": 0
# }'

#{
#  "detail": [
#    {
#      "type": "literal_error",
#      "loc": [
#        "body",
#        "model"
#      ],
#      "msg": "Input should be 'tinyllama' or 'gemma2b'",
#      "input": "gpt-4o",
#      "ctx": {
#        "expected": "'tinyllama' or 'gemma2b'"
#      }
#    }
#  ]
#}


# --------------------------------
# Model validations
# --------------------------------


# schemas.py

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveInt,
    validate_call,
)

ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]
SupportedModels = Annotated[
    Literal["tinysd", "sd1.5"], "Supported Image Generation Models"
]


@validate_call
def is_square_image(value: ImageSize) -> ImageSize:
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value


@validate_call
def is_valid_inference_step(
    num_inference_steps: int, model: SupportedModels
) -> int:
    if model == "tinysd" and num_inference_steps > 2000:
        raise ValueError(
            "TinySD model cannot have more than 2000 inference steps"
        )
    return num_inference_steps


OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)]
InferenceSteps = Annotated[
    int,
    AfterValidator(
        lambda v, values: is_valid_inference_step(v, values["model"])
    ),
]


class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]


class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize
    num_inference_steps: InferenceSteps = 200

# --------------------------------
# Compute field
# --------------------------------

# schemas.py

from typing import Annotated
from pydantic import computed_field, Field
from utils import count_tokens

...


class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]

    @property
    @computed_field
    def tokens(self) -> int:
        return count_tokens(self.content)

    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens
    
response = TextModelResponse(content="FastAPI Generative AI Service", ip=None)
response.model_dump(exclude_none=True)

# {'content': 'FastAPI Generative AI Service',
#  'cost': 0.06,
#  'created_at': datetime.datetime(2024, 3, 7, 20, 42, 38, 729410),
#  'price': 0.01,
#  'request_id': 'a3f18d85dcb442baa887a505ae8d2cd7',
#  'tokens': 6}

response.model_dump_json(exclude_unset=True)
# '{"ip":null,"content":"FastAPI Generative AI Service","tokens":6,"cost":0.06}'


# --------------------------------
# Settings
# --------------------------------


# settings.py

from typing import Annotated

from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[
        PostgresDsn,
        Field(
            alias="DATABASE_URL",
            default="postgres://user:pass@localhost:5432/database",
        ),
    ]
    cors_whitelist_domains: Annotated[
        set[HttpUrl],
        Field(alias="CORS_WHITELIST", default=["http://localhost:3000"]),
    ]


settings = AppSettings()
print(settings.model_dump())
"""
{'port': 8000
 'app_secret': 'asdlkajdlkajdklaslkldjkasldjkasdjaslk',
 'pg_dsn': MultiHostUrl('postgres://sa:password@localhost:5432/cms'),
 'cors_whitelist_domains': {Url('http://localhost:3000/'),
                            Url('https://xyz.azurewebsites.net/')},
}
"""

# --------------------------------
# Dataclasses in FastAPI
# --------------------------------


# schemas.py

from dataclasses import dataclass
from typing import Literal


@dataclass
class TextModelRequest:
    model: Literal["tinyLlama", "gemma2b"]
    prompt: str
    temperature: float


@dataclass
class TextModelResponse:
    response: str
    tokens: int


# main.py

from fastapi import Body, FastAPI, HTTPException, status
from models import generate_text, load_text_model
from schemas import TextModelRequest, TextModelResponse
from utils import count_tokens

# load lifespan
...

app = FastAPI(lifespan=lifespan)


@app.post("/generate/text")
def serve_text_to_text_controller(
    body: TextModelRequest = Body(...),
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]:
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    tokens = count_tokens(body.prompt) + count_tokens(output)
    return TextModelResponse(response=output, tokens=tokens)


# --------------------------------
# Pydantic usage in FastAPI
# --------------------------------

# main.py

from fastapi import Body, FastAPI, HTTPException, Request, status
from models import generate_text
from schemas import TextModelRequest, TextModelResponse

# load lifespan
...

app = FastAPI(lifespan=lifespan)


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