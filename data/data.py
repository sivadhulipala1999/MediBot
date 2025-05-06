from datasets import load_dataset


class Data:
    def __init__(self, dataset_path, tokenizer, device):
        self.tokenizer = tokenizer

        self.device = device

        self.dataset = load_dataset(dataset_path)
        self.dataset = self.dataset["train"].train_test_split(test_size=0.1)

        self.tokenized_dataset = self.dataset.map(
            self.tokenize_function, batched=True)

    def fetch_tokenized_data(self):
        return self.tokenized_dataset

    def tokenize_function(self, examples):
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # We format the dataset as per the chat template
        user_prompt = [f"""<|system|>
    You are a helpful medical assistant. Answer questions clearly and concisely.
    <|user|>
    {q}
    <|assistant|>
    """
                       for q in examples["question"]
                       ]

        answer = [f" {str(a or '')}" for a in examples["answer"]]
        texts = [user_prompt[i] + answer[i] for i in range(len(user_prompt))]

        # Tokenize all texts in the batch
        inputs = self.tokenizer(texts, padding="max_length",
                                truncation=True, max_length=1000).to(self.device)

        labels = inputs["input_ids"].copy()

        # Prepare the labels
        for i in range(len(inputs["input_ids"])):
            user_prompt_ids = self.tokenizer(
                user_prompt[i], truncation=True, add_special_tokens=False)["input_ids"]
            # Length of user prompt tokens
            prompt_length = len(user_prompt_ids)
            labels[i][:prompt_length] = [self.tokenizer.pad_token_id] * \
                prompt_length  # Mask out the prompt tokens

        inputs["labels"] = labels
        return inputs
