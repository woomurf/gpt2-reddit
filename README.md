# GPT2 Reddit

API server provides gpt2 api that fine tuned by reddit dataset. gpt2 model are from [huggingface](https://huggingface.co/mrm8488/gpt2-finetuned-reddit-tifu). 

### Install 
```bash
$ pip install -r requirements.txt
```

### Usage 
```bash
$ python3 server.py
```
and API server hosted on `localhost:8000`

### Docker
```bash
$ docker build -t ${IMAGE_NAME} .

$ docker run -p ${YOUR_PORT}:8000 -it ${IMAGE_NAME}
```

### cURL
```bash
$ curl -X POST "https://master-gpt2-reddit-woomurf.endpoint.ainize.ai/gpt2-reddit/short" \
  -H "accept: application/json" -H "Content-Type: multipart/form-data" \
  -F "text=test input" -F "num_samples=5"
```

or 

```bash
$ curl -X POST "https://master-gpt2-reddit-woomurf.endpoint.ainize.ai/gpt2-reddit/long" \
  -H "accept: multipart/form-data" -H "Content-Type: multipart/form-data"\
  -F "text=test input" -F "num_samples=5" -F "length=5"
```