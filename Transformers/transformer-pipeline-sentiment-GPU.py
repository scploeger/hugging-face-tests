from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

## without PyTorch 
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

clsf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = ["This product is great!",
           "I would not reccomend this product to my enemies...",
           "I like rice and beans for dinner.",
           "We already know the truth.",
           "The recipe says to use one tablespoon.",
           "His pencil case was filled entirely with Cheez-its.",
           "He didn't know where he was supposed to go for his interview.",
           "I would not reccomend this product to my enemies...",
           "I like rice and beans for dinner.",
           "We already know the truth.",
           "The recipe says to use one tablespoon.",
           "His pencil case was filled entirely with Cheez-its.",
           "He didn't know where he was supposed to go for his interview.",
           "I would not reccomend this product to my enemies...",
           "I like rice and beans for dinner.",
           "We already know the truth.",
           "The recipe says to use one tablespoon.",
           "His pencil case was filled entirely with Cheez-its.",
           "He didn't know where he was supposed to go for his interview.",
           "I would not reccomend this product to my enemies...",
           "I like rice and beans for dinner.",
           "We already know the truth.",
           "The recipe says to use one tablespoon.",
           "His pencil case was filled entirely with Cheez-its.",
           "He didn't know where he was supposed to go for his interview."]

res = clsf(X_train)
print(res)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

## with PyTorch
batch = tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
model = model.to(device)

with torch.no_grad(): # make an inference with PyTorch
        outputs = model(**batch) # unpack the batch to get single examples from input dict
        print(outputs)
        preds = F.softmax(outputs.logits, dim=1) # use softmax to get preds
        print(preds)
        labels = torch.argmax(preds, dim=1) # argmax to get labels
        print(labels)