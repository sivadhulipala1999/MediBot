# MediBot
Fine-tuning is more relevant than ever before since we have the power of taming large language models to answer questions specific to our domains. I built this notebook to gain a better understanding of the current landscape of fine-tuning and share my knowledge with everyone else! 

In the notebook, you will find a fine-tuned TinyLlama model, that answers medical questions. MedQuAD was the dataset used and LoRA was chosen as the main approach to do the fine-tuning. Please consider this as a small scale proof-of-concept to understand how domain adaptation of LLMs could work. 

Update 05/2025: Following the fine-tuning, new evaluation scores (exact match and BERTScore) were added to test the model's performance in a "post-production" setting. Along with this, a streamlit frontend was added to give a simple UI for interaction. 
