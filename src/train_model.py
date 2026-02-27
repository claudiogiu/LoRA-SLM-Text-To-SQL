import os
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from peft import (
    LoraConfig,
    get_peft_model
)

from preprocess import (
    WikiSQLSeq2SeqFormatter,
    WikiSQLCausalLMInstructionFormatter,
    WikiSQLCausalLMChatFormatter
)

warnings.filterwarnings("ignore")

@dataclass
class WikiSQLT5LoRATrainer:
    """
    Interface for fine-tuning models in the T5 family on the WikiSQL dataset
    using standard FP16 LoRA adaptation within an encoder–decoder (seq2seq) workflow.

    Attributes:
        model_name (str): Name or path of the pretrained T5 model to load.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test').

        _setup_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
            Initializes the tokenizer, loads the selected T5 model in FP16 mode,
            enables gradient checkpointing, and injects LoRA adapters into
            attention and feed-forward projection layers.

        _preprocess(examples: dict, tokenizer) -> dict:
            Tokenizes natural-language questions and SQL targets, applies padding
            and truncation, and masks padding tokens in the label sequences.

        _save_model(model, tokenizer) -> None:
            Stores the fine-tuned model and tokenizer in the 'models/' directory.

        train() -> None:
            Executes the complete fine-tuning pipeline.
    
    """

    model_name: str = "t5-base"

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLSeq2SeqFormatter(split=split)
        return processor.to_dataset()

    def _setup_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA trainable params: {trainable_params} / {total_params} "
            f"({100 * trainable_params / total_params:.4f}% of total)"
        )

        return model, tokenizer
    
    def _preprocess(self, examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["input"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

        labels = tokenizer(
            text_target=examples["output"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

        labels_ids = []
        for seq in labels["input_ids"]:
            labels_ids.append([
                -100 if token_id == tokenizer.pad_token_id else token_id
                for token_id in seq
            ])

        model_inputs["labels"] = labels_ids
        return model_inputs

    def _save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'T5')
        os.makedirs(models_dir, exist_ok=True)
        model.save_pretrained(models_dir)
        tokenizer.save_pretrained(models_dir)
        print(f"Model and tokenizer successfully saved to: {models_dir}")

    def train(self) -> None:
        train_dataset = self._load_dataset("train")
        dev_dataset = self._load_dataset("dev")

        model, tokenizer = self._setup_model()

        train_tokenized = train_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )
        dev_tokenized = dev_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )

        args = Seq2SeqTrainingArguments(
            output_dir="./results",
            eval_strategy="no",
            logging_steps=200,
            per_device_train_batch_size=16, 
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            optim="adamw_torch",
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_tokenized,
            eval_dataset=dev_tokenized
        )

        print("Starting training...")
        trainer.train()

        self._save_model(model, tokenizer)


@dataclass
class WikiSQLQwenLoRATrainer:
    """
    Interface for fine-tuning models in the Qwen2.5 family on the WikiSQL dataset
    using standard FP16 LoRA adaptation within a decoder-only (causal LM) workflow.

    Attributes:
        model_name (str): Name or path of the pretrained Qwen2.5 model to load.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test').

        _setup_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
            Initializes the tokenizer, loads the selected Qwen model in FP16 mode,
            enables gradient checkpointing, and injects LoRA adapters into
            attention and MLP projection layers.

        _preprocess(examples: dict, tokenizer) -> dict:
            Tokenizes instruction-style prompts, applies padding and truncation,
            and masks all tokens preceding the SQL answer segment.

        _save_model(model, tokenizer) -> None:
            Stores the fine-tuned model and tokenizer in the 'models/' directory.

        train() -> None:
            Executes the complete fine-tuning pipeline.
    
    """

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLCausalLMInstructionFormatter(split=split)
        return processor.to_dataset()

    def _setup_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA trainable params: {trainable_params} / {total_params} "
            f"({100 * trainable_params / total_params:.4f}% of total)"
        )

        return model, tokenizer

    def _preprocess(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=192
        )

        input_ids = torch.tensor(model_inputs["input_ids"])
        labels = input_ids.clone()

        sql_token_ids = tokenizer.encode("### SQL:", add_special_tokens=False)

        for i, seq in enumerate(input_ids):
            pos = None

            for j in range(len(seq) - len(sql_token_ids)):
                if seq[j:j+len(sql_token_ids)].tolist() == sql_token_ids:
                    pos = j + len(sql_token_ids)
                    break

            if pos is not None:
                labels[i, :pos] = -100

            labels[i, labels[i] == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels.tolist()
        return model_inputs

    def _save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'Qwen')
        os.makedirs(models_dir, exist_ok=True)
        model.save_pretrained(models_dir)
        tokenizer.save_pretrained(models_dir)
        print(f"Model and tokenizer successfully saved to: {models_dir}")

    def train(self) -> None:
        train_dataset = self._load_dataset("train")
        dev_dataset = self._load_dataset("dev")

        model, tokenizer = self._setup_model()

        train_tokenized = train_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )
        dev_tokenized = dev_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )

        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="no",
            logging_steps=200,
            per_device_train_batch_size=16, 
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            optim="adamw_torch",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tokenized,
            eval_dataset=dev_tokenized
        )

        print("Starting training...")
        trainer.train()

        self._save_model(model, tokenizer)


@dataclass
class WikiSQLLlamaLoRATrainer:
    """
    Interface for fine-tuning models in the Llama 3.x family on the WikiSQL dataset
    using standard FP16 LoRA adaptation within a decoder-only (causal LM) workflow.

    Attributes:
        model_name (str): Name or path of the pretrained Llama 3.x model to load.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test').

        _setup_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
            Initializes the tokenizer, loads the selected Llama model in FP16 mode,
            enables gradient checkpointing, and injects LoRA adapters into
            attention projection layers.

        _preprocess(examples: dict, tokenizer) -> dict:
            Tokenizes chat-style prompts, applies padding and truncation,
            and masks all tokens preceding the SQL answer segment.

        _save_model(model, tokenizer) -> None:
            Stores the fine-tuned model and tokenizer in the 'models/' directory.

        train() -> None:
            Executes the complete fine-tuning pipeline.
    
    """

    model_name: str = "meta-llama/Llama-3.2-3B"

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLCausalLMChatFormatter(split=split)
        return processor.to_dataset()

    def _setup_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=False
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA trainable params: {trainable_params} / {total_params} "
            f"({100 * trainable_params / total_params:.4f}% of total)"
        )

        return model, tokenizer

    def _preprocess(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=192
        )

        input_ids = torch.tensor(model_inputs["input_ids"])
        labels = input_ids.clone()

        sql_token_ids = tokenizer.encode("### Response:", add_special_tokens=False)

        for i, seq in enumerate(input_ids):
            pos = None
            for j in range(len(seq) - len(sql_token_ids)):
                if seq[j:j+len(sql_token_ids)].tolist() == sql_token_ids:
                    pos = j + len(sql_token_ids)
                    break

            if pos is not None:
                labels[i, :pos] = -100

            labels[i, labels[i] == tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels.tolist()
        return model_inputs

    def _save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'Llama')
        os.makedirs(models_dir, exist_ok=True)
        model.save_pretrained(models_dir)
        tokenizer.save_pretrained(models_dir)
        print(f"Model and tokenizer saved to: {models_dir}")

    def train(self) -> None:
        train_dataset = self._load_dataset("train")
        dev_dataset = self._load_dataset("dev")

        model, tokenizer = self._setup_model()

        train_tokenized = train_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )
        dev_tokenized = dev_dataset.map(
            lambda e: self._preprocess(e, tokenizer),
            batched=True
        )

        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="no",
            logging_steps=200,
            per_device_train_batch_size=16, 
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            optim="adamw_torch",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tokenized,
            eval_dataset=dev_tokenized
        )

        print("Starting training...")
        trainer.train()

        self._save_model(model, tokenizer)


if __name__ == "__main__":
    print("\nT5-Base — Training Session\n")
    t5_trainer: WikiSQLT5LoRATrainer = WikiSQLT5LoRATrainer()
    t5_trainer.train()

    print("\nQwen2.5-3B-Instruct — Training Session\n")
    qwen_trainer: WikiSQLQwenLoRATrainer = WikiSQLQwenLoRATrainer()
    qwen_trainer.train()

    print("\nLLama-3.2-3B — Training Session\n")
    llama_trainer: WikiSQLLlamaLoRATrainer = WikiSQLLlamaLoRATrainer()
    llama_trainer.train()