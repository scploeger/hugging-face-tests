from transformers import pipeline # pipelines group pretrained models together with the matching preprocessing

clsf = pipeline("sentiment-analysis") # list of piplines here: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task

res = clsf("Wow, this is really great!") # uses the pipeline to classify this text
print(res)

res = clsf("This product is super bad!") # uses the pipeline to classify this text
print(res)

# test to do some of the pipeline steps manually
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

clsf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) # list of piplines here: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task

res = clsf("Wow, this is really great!") # uses the pipeline to classify this text
print(res)