FROM python:3.9

COPY requirements.txt /requirements.txt
COPY api /api
COPY xray /xray
COPY utils /utils
COPY models/ models/

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
