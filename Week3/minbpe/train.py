import os
import time

from minbpe.basic import BasicTokenizer

if __name__=="__main__":

    # cd to current .py directyory
    target_dir = "./Week3/minbpe"
    os.chdir(target_dir)

    # create a directory for models, so we don't pollute the current directory
    os.makedirs("models", exist_ok=True)

    # open some text and train a vocab of 512 tokens
    text = open("./test/taylorswift.txt", "r", encoding="utf-8").read()

    t0 = time.time()

    tokenizer = BasicTokenizer()
    tokenizer.train(text, 512, verbose=False)
    
    # text = "hello world!I love you because I really think you are the best and you are the best gift I can ever get in my life. I hope I can stay with you in my whole life and I will use out of my love and treat you as my little baby"
    # tokens = tokenizer.encode(text)
    # new_text = tokenizer.decode(tokens)
    # print(tokens)
    # print(new_text)
    # print(text==new_text)

    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", "basic")
    tokenizer.save(prefix)

    t1 = time.time()

    print(f"Training took {t1 - t0:.2f} seconds")