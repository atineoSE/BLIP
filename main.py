import io
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from api.captioner import Captioner
from api.prompts import style_question
from api.questioner import Questioner

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

load_dotenv()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Setting up captioner with image size {IMAGE_SIZE}")
    app.captioner = Captioner(image_size=IMAGE_SIZE)
    app.questioner = Questioner(image_size=IMAGE_SIZE)
    yield
    # Clean up, if needed
    pass


app = FastAPI(title="BLIP", description="BLIP for image captioning", lifespan=lifespan)


@app.post("/caption")
async def get_caption(request: Request, image: UploadFile) -> JSONResponse:
    contents = await image.read()
    input_image = Image.open(io.BytesIO(contents))
    caption = request.app.captioner.get_caption(input_image)
    return JSONResponse(caption)


@app.post("/answer")
async def get_answer(
    request: Request, image: UploadFile, question: str
) -> JSONResponse:
    contents = await image.read()
    input_image = Image.open(io.BytesIO(contents))
    answer = request.app.questioner.get_answer(input_image, question)
    return JSONResponse(answer)


@app.post("/rich_caption")
async def get_rich_caption(request: Request, image: UploadFile) -> JSONResponse:
    contents = await image.read()
    input_image = Image.open(io.BytesIO(contents))
    caption = request.app.captioner.get_caption(input_image)
    style = request.app.questioner.get_answer(input_image, style_question)
    rich_caption = caption + ", " + style
    return JSONResponse(rich_caption)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
