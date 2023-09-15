from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Thank you for submitting your application for the Analyst, Master Data Management position at our company.
After reviewing your profile, we regret to inform you that your application was not selected for the next steps of our hiring process for the role you applied for.
"""

print(summarizer(text, max_length=30, min_length=10, do_sample=False))