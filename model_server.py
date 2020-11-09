from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading
import torch
import json

from util import get_bad_word_list

app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

model = AutoModelWithLMHead.from_pretrained("mrm8488/gpt2-finetuned-reddit-tifu", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

bad_word_tokens = get_bad_word_list()

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                requests['output'] = run_generate(requests['input'][0], requests['input'][1], requests['input'][2])


threading.Thread(target=handle_requests_by_batch).start()

def run_word(input_ids):
    token_tensor = torch.LongTensor([input_ids]).to(device)

    output = model(token_tensor)

    output = output.to_tuple()

    return {'output': output} 

def run_generate(input_ids, num_samples, length):
    inputs = []

    for input_id in input_ids:
        inputs.append(int(input_id))

    token_tensor = torch.LongTensor([inputs]).to(device)

    outputs = model.generate(
        token_tensor,
        pad_token_id=50256,
        max_length=length,
        min_length=length,
        do_sample=True,
        top_k=50,
        num_return_sequences=num_samples,
        bad_words_ids=bad_word_tokens
    )

    outputs = outputs.tolist()

    return {"output": outputs}

@app.route("/gpt2-reddit/<type>", methods=["POST"])
def gpt2(type):
    if type not in ["short", "long"]:
        return jsonify({'error': 'This is wrong address'}), 404
    
    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429

    data = request.json

    try:
        args= [] 
        args.append(data['input_ids'])
        if type == "long":
            args.append(data['num_samples'])
            args.append(data['length'])
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008)