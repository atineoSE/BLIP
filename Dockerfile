FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get -y update && apt-get install -y python3-pip

RUN mkdir /blip
WORKDIR /blip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
