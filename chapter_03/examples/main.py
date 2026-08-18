from io import BytesIO

from fastapi import FastAPI, File, Response, status
from fastapi.responses import StreamingResponse
from models import (
    generate_3d_geometry,
    generate_audio,
    generate_image,
    generate_text,
    generate_video,
    load_3d_model,
    load_audio_model,
    load_image_model,
    load_text_model,
    load_video_model,
)
from PIL import Image
from schemas import VoicePresets
from utils import audio_array_to_buffer, export_to_video_buffer, img_to_bytes, mesh_to_obj_buffer
import httpx
from fastapi import FastAPI, Response
from fastapi import FastAPI
from openai import OpenAI
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()


@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    pipe = load_text_model()
    output = generate_text(pipe, prompt)
    return output


@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")


@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
def serve_text_to_image_model_controller(prompt: str):
    pipe = load_image_model()
    output = generate_image(pipe, prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")


@app.post(
    "/generate/video",
    responses={status.HTTP_200_OK: {"content": {"video/mp4": {}}}},
    response_class=StreamingResponse,
)
def serve_image_to_video_model_controller(image: bytes = File(...), num_frames: int = 25):
    image = Image.open(BytesIO(image))
    model = load_video_model()
    frames = generate_video(model, image, num_frames)
    return StreamingResponse(export_to_video_buffer(frames), media_type="video/mp4")


@app.get(
    "/generate/3d",
    responses={status.HTTP_200_OK: {"content": {"model/obj": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_3d_model_controller(prompt: str, num_inference_steps: int = 25):
    model = load_3d_model()
    mesh = generate_3d_geometry(model, prompt, num_inference_steps)
    response = StreamingResponse(mesh_to_obj_buffer(mesh), media_type="model/obj")
    response.headers["Content-Disposition"] = f"attachment; filename={prompt}.obj"
    return response


@app.get(
    "/generate/bentoml/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def serve_bentoml_text_to_image_controller(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
        "http://localhost:5001/generate", json={"prompt": prompt}
        )
    return Response(content=response.content, media_type="image/png")


load_dotenv()
API_KEY = os.environ["OPENAI_API__KEY"]
MODEL = os.environ["OPENAI_API__MODEL"]
BASE_URL = os.environ["OPENAI_API__BASE"]
openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
system_prompt = "You are a helpful assistant."

@app.get("/generate/openai/text")
def serve_openai_language_model_controller(prompt: str) -> str | None:
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content