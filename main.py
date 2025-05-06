from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import utils.utils as utils
from data.data import Data
from evaluation.eval_methods import EvalBenchmark


def main():
    config = utils.load_config()

    # Authenticate with Read Token (to download model)
    login(token=config["HF_TOKEN"])

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(config['MODEL_NAME'])
    model.to(device)  # Put the model on GPU if available
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])

    # Tokenized Dataset
    dataset = Data(dataset_path="lavita/MedQuAD",
                   tokenizer=tokenizer, device=device)

    # Evaluate Model
    eval_benchmark = EvalBenchmark(model, dataset)
    eval_benchmark.show_metrics()


if __name__ == "__main__":
    main()
