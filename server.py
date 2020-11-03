from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading
import torch

from util import get_bad_word_list

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

tokenizer = AutoTokenizer.from_pretrained("mrm8488/gpt2-finetuned-reddit-tifu")
model = AutoModelWithLMHead.from_pretrained("mrm8488/gpt2-finetuned-reddit-tifu", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                if len(requests['input']) == 2:
                    requests['output'] = run_word(requests['input'][0], requests['input'][1])
                else:
                    requests['output'] = run_generate(requests['input'][0], requests['input'][1], requests['input'][2])


# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()

def run_word(sequence, num_samples):
    print("word!")
    input_ids = tokenizer.encode(sequence, return_tensors="pt")
    tokens_tensor = input_ids.to(device)
    next_token_logits = model(tokens_tensor).logits[:, -1, :]
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=num_samples)

    result = dict()
    for idx, token in enumerate(next_token.tolist()[0]):
        result[idx] = tokenizer.decode(token)
    print(result)
    return result

def run_generate(text, num_samples, length):
    print("generate!")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens_tensor = input_ids.to(device)
    min_length = len(input_ids.tolist()[0])
    length += min_length
    
    bad_word_tokens = get_bad_word_list()

    outputs = model.generate(tokens_tensor,
        pad_token_id=50256, 
        max_length=length, 
        min_length=length, 
        do_sample=True, 
        top_k=num_samples,
        num_return_sequences=num_samples,
        bad_word_ids=bad_word_tokens)

    result = {}
    for idx, output in enumerate(outputs):
        result[idx] = tokenizer.decode(output.tolist()[min_length:], skip_special_tokens=True)
        
    print(result)
    return result

@app.route("/gpt2-reddit/<mode>", methods=['POST'])
def run_gpt2_reddit(mode):
    if mode not in ["short", "long"]:
        return jsonify({'error': 'This is wrong address'}), 400

    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429

    # 웹페이지로부터 이미지와 스타일 정보를 얻어옴.
    try:
        args = []
        args.append(request.form['text'])
        args.append(int(request.form['num_samples']))
        if mode == "long":
            length = args.append(int(request.form['length']))

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    # Queue - put data
    req = {
        'input': args
    }
    requests_queue.put(req)

    # Queue - wait & check
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    return result

# Health Check
@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)