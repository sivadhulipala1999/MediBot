"""Some benchmarking metrics"""

import torch
from bert_score import score


class EvalBenchmark:
    def __init__(self, model, dataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = dataset.tokenizer

        self.model = model
        self.inputs = [f"""<|system|> 
            You are a helpful medical assistant. Answer questions clearly and concisely.
            <|user|>
            {q}
            <|assistant|>""" for q in dataset.dataset["train"]["question"][-100:]]
        self.labels = [
            f" {str(a or '')}" for a in dataset.dataset["train"]["answer"][-100:]]

        self.max_new_tokens = 100
        self.gen_outputs = self._generate(self.inputs)

    def _generate(self, prompts):
        if isinstance(prompts, list):
            outputs = []
            for prompt in prompts:
                input = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True).to(self.device)
                output = self.model.generate(
                    **input, max_new_tokens=self.max_new_tokens)
                outputs.append(self.tokenizer.decode(
                    output[0], skip_special_tokens=True).split("<|assistant|>")[1])
        else:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", truncation=True).to(self.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens)
        return outputs

    def exact_match(self):
        correct_count = 0
        for i, d in enumerate(self.gen_outputs):
            predict_text = self.gen_outputs[i]
            golden = self.labels[i]
            if golden == predict_text:
                correct_count += 1
        return correct_count

    def bert_score(self):
        P, R, F1 = score(self.gen_outputs, self.labels,
                         lang="en", verbose=True)
        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item()
        }

    def show_metrics(self):
        print(f"Exact Match: {self.exact_match()}")
        print(f"BERT Score: {self.bert_score()}")
