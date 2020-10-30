FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app 

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . . 

CMD ["python3", "server.py"]