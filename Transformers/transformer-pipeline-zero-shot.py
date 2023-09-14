from transformers import pipeline

gen = pipeline("zero-shot-classification") # list of piplines here: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task

res = gen("The puppy reported lost earlier today was found safe.",
          candidate_labels=["education", "review", "science", "news"]
          )

print(res)

res = gen("The periodic table symbol for potassium is K.",
          candidate_labels=["education", "review", "science", "news"]
          )

print(res)