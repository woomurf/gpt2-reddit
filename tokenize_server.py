from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from flask import Flask, request, Response, jsonify
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading
import torch

import json
from util import get_bad_word_list
import numpy as np
import requests

app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1


models = {
    "gpt2-large": "http://main-gpt2-large-jeong-hyun-su.endpoint.ainize.ai/",
    "gpt2-cover-letter": "http://main-gpt2-cover-letter-jeong-hyun-su.endpoint.ainize.ai/",
    "gpt2-reddit": "http://localhost:8008/",
    "gpt2-story": "http://main-gpt2-story-gmlee329.endpoint.ainize.ai/",
    "gpt2-ads": "http://main-gpt2-ads-psi1104.endpoint.ainize.ai/",
    "gpt2-business": "http://main-gpt2-business-leesangha.endpoint.ainize.ai/",
    "gpt2-film": "http://main-gpt2-film-gmlee329.endpoint.ainize.ai/",
    "gpt2-trump": "http://main-gpt2-trump-gmlee329.endpoint.ainize.ai/"
}

tokenizer_url = {
    "gpt2-reddit": "mrm8488/gpt2-finetuned-reddit-tifu",
}

def load_tokenizers():
    tokenizers = {}
    
    for name, url in tokenizer_url.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(url)

    return tokenizers

tokenizers = load_tokenizers()

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:    
                requests['output'] = run_generation(requests['input'][0], requests['input'][1], requests['input'][2])

threading.Thread(target=handle_requests_by_batch).start()

def run_word(context, length, model):
    tokenizer = tokenizers[model]
    input_ids = tokenizer.encode(context)

    URL = models[model] + model + "/" + length
    data = {
        "input_ids": input_ids,
    }

    response = request.post(URL, data=data)

    if response.status_code != 200:
        return {'error':'error'} # Need idea about error message 
    
    print(response.json())
    next_token_logits = response.json()['output'][1]
    
    # Need data type change #######
    #
    #
    ###############################

    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=num_samples)

    result = dict()
    for idx, token in enumerate(next_token.tolist()[0]):
        result[idx] = tokenizer.decode(token)

    return result

def run_generation(model, text, mode):
    tokenizer = tokenizers[model]
    input_ids = tokenizer.encode(text)

    min_length = len(input_ids)
    length = min_length + 20

    url = models[model] + model + "/" + mode

    data = json.dumps({
        "input_ids": input_ids,
        "num_samples": 5,
        "length": length
    })

    header = {"content-type":"application/json"}
    response = requests.post(url, data=data, headers=header)

    if response.status_code != 200:
        return {"error":"error"}
    
    outputs = response.json()['output']

    result = {}
    for idx, output in enumerate(outputs):
        result[idx] = tokenizer.decode(output[min_length:], skip_special_tokens=True)

    return result

@app.route("/gpt2", methods=["POST"])
def gpt2():
    
    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429
    
    try:
        args = []

        model = request.form['model']
        context = request.form['context']
        length = request.form['length']

        args = [model, context, length]
    except Exception:
        return jsonify({'error':'Invalid Inputs'}), 400

    req = {
        'input': args
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    
    result = req['output']

    return result

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)