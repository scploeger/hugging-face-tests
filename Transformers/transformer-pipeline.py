from transformers import pipeline # pipelines group pretrained models together with the matching preprocessing

clsf = pipeline("sentiment-analysis") # downloads and caches the the pretrained model used by the pipeline

res = clsf("Wow, this is really great!") # uses the pipeline to classify this text

print(res)

res = clsf("This product is super bad!") # uses the pipeline to classify this text

print(res)