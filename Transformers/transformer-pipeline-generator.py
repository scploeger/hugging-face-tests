from transformers import pipeline # pipelines group pretrained models together with the matching preprocessing

gen = pipeline("text-generation", model="distilgpt2") # example of passing a specific model (from model hub)

res = gen("The quick brown fox jumps",
          max_length = 20,
          num_return_sequences=2,
          )

print(res)

res = gen("When I am bored I like to",
          max_length = 20,
          num_return_sequences=2,
          )

print(res)