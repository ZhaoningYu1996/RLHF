import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from datasets import Dataset, DatasetDict
import pandas as pd

class Config:
    """Configuration for the SFT training process."""
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-1B" 
        self.train_file = "datasets/fairseq2-lm-gsm8k/sft/train.jsonl"
        self.test_file = "datasets/fairseq2-lm-gsm8k/test/test.jsonl" 
        self.output_dir = "llama3-gsm8k-sft"
        self.logging_dir = "llama3-gsm8k-sft-logs"

        # LoRA Configuration
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training Arguments (integrated)
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            gradient_accumulation_steps=6,
            learning_rate=2e-5,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            report_to="tensorboard",
            # gradient_checkpointing=True,  # Enable if needed
        )

        self.max_seq_length = 512
        self.special_tokens = {'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

class DataLoader:
    """Handles loading, formatting, and tokenization of the dataset."""
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.add_special_tokens(self.config.special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_jsonl(self, filepath):
        """Loads data from a JSONL file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def formatting_prompts_func(self, examples):
        """Formats prompts."""
        src_list = examples['src']
        tgt_list = examples['tgt']
        return {"text": [f"{src}{tgt}<|eot_id|>" for src, tgt in zip(src_list, tgt_list)]}

    def load_and_process_data(self):
        """Loads, preprocesses, and tokenizes data."""
        train_data, test_data = self.load_jsonl(self.config.train_file), self.load_jsonl(self.config.test_file)
        train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(data=test_data))
        train_dataset = train_dataset.map(self.formatting_prompts_func, batched=True, remove_columns=["src", "tgt"])
        test_dataset = test_dataset.map(self.formatting_prompts_func, batched=True, remove_columns=["src", "tgt"])

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.config.max_seq_length)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        return DatasetDict({"train": train_dataset, "test": test_dataset})

class ModelTrainer:
    """Manages model initialization, LoRA preparation, and training."""
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        """Initializes the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            # load_in_8bit=True, # Or 4bit
        )
        self.model = self.model.to("cpu")
        self.model.resize_token_embeddings(len(DataLoader(self.config).tokenizer))
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = DataLoader(self.config).tokenizer.eos_token_id
        self.model = self.model.to("cuda")

    def prepare_lora(self):
        """Prepares the model for LoRA."""
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.config.lora_config)  # Use config.lora_config
        self.model.print_trainable_parameters()

    def train(self, dataset_dict):
        """Trains the model."""
        print(f"Effective Train Batch Size: {self.config.training_args.per_device_train_batch_size * self.config.training_args.gradient_accumulation_steps}")
        print(f"Max Sequence Length: {self.config.max_seq_length}")
        print(f"Train Dataset Length: {len(dataset_dict['train'])}")
        print(f"Eval Dataset Length: {len(dataset_dict['test'])}")
        data_collator = DataCollatorForLanguageModeling(tokenizer=DataLoader(self.config).tokenizer, mlm=False)
        trainer = SFTTrainer(
            model=self.model,
            args=self.config.training_args,  # Use config.training_args
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model(self.config.output_dir)
        DataLoader(self.config).tokenizer.save_pretrained(self.config.output_dir)

    def inference(self, prompt):
        # 1. Load the tokenizer (from the output directory)
        tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)

        # 2. Load the *base* model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # load_in_8bit=True # If you used 8-bit during training
        )

        # 3. *Resize embeddings* to match the tokenizer (CRITICAL STEP)
        model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

        # 4. Load the *PEFT model*
        model = PeftModel.from_pretrained(model, self.config.output_dir)
        model.eval()


        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

def main():
    config = Config()
    data_loader = DataLoader(config)
    dataset_dict = data_loader.load_and_process_data()
    model_trainer = ModelTrainer(config)
    model_trainer.initialize_model()
    model_trainer.prepare_lora()
    model_trainer.train(dataset_dict)
    prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    print(model_trainer.inference(prompt))

if __name__ == "__main__":
    main()