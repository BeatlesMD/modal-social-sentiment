"""
QLoRA fine-tuning for Qwen3-4B on Modal.

Uses PEFT for efficient fine-tuning on a single A100.
"""

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer.
    
    Configures and runs fine-tuning using:
    - 4-bit quantization (QLoRA)
    - LoRA adapters
    - SFTTrainer from trl
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        output_dir: str = "/models/fine-tuned",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LoRA config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Training config
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        
        self.logger = logger.bind(component="finetune")
    
    def train(self, train_data_path: str, val_data_path: str | None = None):
        """
        Run fine-tuning.
        
        Args:
            train_data_path: Path to training JSONL file
            val_data_path: Optional path to validation JSONL file
        """
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        
        self.logger.info("Starting QLoRA fine-tuning", model=self.base_model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Quantization config for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        self.logger.info("Loading base model with 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load dataset
        self.logger.info("Loading training data", path=train_data_path)
        
        data_files = {"train": train_data_path}
        if val_data_path:
            data_files["validation"] = val_data_path
        
        dataset = load_dataset("json", data_files=data_files)
        
        # Format function for chat template
        def format_chat(example):
            return {"text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )}
        
        dataset = dataset.map(format_chat)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            evaluation_strategy="steps" if val_data_path else "no",
            eval_steps=100 if val_data_path else None,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",  # Set to "wandb" if using W&B
            optim="paged_adamw_8bit",
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            packing=False,
        )
        
        # Train
        self.logger.info("Starting training")
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final"
        self.logger.info("Saving model", path=str(final_model_path))
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save training config
        config = {
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
        }
        with open(final_model_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("Training complete!")
        return str(final_model_path)


def merge_lora_weights(
    base_model: str,
    adapter_path: str,
    output_path: str,
):
    """
    Merge LoRA weights into base model for faster inference.
    
    This creates a full model that can be loaded without PEFT.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info("Merging LoRA weights", base=base_model, adapter=adapter_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load and merge adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Merged model saved", path=output_path)
    return output_path
