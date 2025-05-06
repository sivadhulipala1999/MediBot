from transformers import AutoModelForCausalLM, AutoTokenizer
import utils.utils as utils
import torch


class MediBot():
    def __init__(self):
        config = utils.load_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(config['MODEL_NAME'])
        self.model.to(self.device)  # Put the model on GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])

        self.max_new_tokens = 100

    def query_llm(self, prompt):
        prompt = f"""<|system|> 
            You are a helpful medical assistant. Answer questions clearly and concisely.
            <|user|>
            {prompt}
            <|assistant|>"""
        input = self.tokenizer(
            prompt, return_tensors="pt", truncation=True).to(self.device)
        output = self.model.generate(
            **input, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(
            output[0], skip_special_tokens=True).split("<|assistant|>")[1]
