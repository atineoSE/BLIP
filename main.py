import io
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from captioner import Captioner

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
    yield
    # Clean up, if needed
    pass


app = FastAPI(title="BLIP", description="BLIP for image captioning", lifespan=lifespan)


@app.post("/caption")
async def get_caption(request: Request, image: UploadFile) -> JSONResponse:
    contents = await image.read()
    caption_image = Image.open(io.BytesIO(contents))
    caption = request.app.captioner.get_caption(image=caption_image)
    return JSONResponse(caption)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
