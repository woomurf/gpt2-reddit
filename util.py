from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/gpt2-finetuned-reddit-tifu")

def get_bad_word_list():
    with open("stop_words.txt") as file:
        bad_words = file.read()

    bad_word_list = bad_words.split("\n")

    bad_word_tokens = []
    for badword in bad_word_list:
        token = tokenizer.encode(badword, add_prefix_space=True)
        bad_word_tokens.append(token)
    
    return bad_word_tokens
    
