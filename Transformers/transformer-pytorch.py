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
           "I would not reccomend this product to my enemies..."]

res = clsf(X_train)
print(res)

## with PyTorch
batch = tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors="pt") # tokenize and return tensors in PyTorch format

with torch.no_grad(): # make an inference with PyTorch
        outputs = model(**batch) # unpack the batch to get single examples from input dict
        print(outputs)
        preds = F.softmax(outputs.logits, dim=1) # use softmax to get preds
        print(preds)
        labels = torch.argmax(preds, dim=1) # argmax to get labels
        print(labels)