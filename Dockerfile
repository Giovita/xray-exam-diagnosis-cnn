FROM python:3.9

COPY requirements.txt /requirements.txt
COPY api /api
COPY xray /xray
COPY utils /utils
COPY models/ models/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
