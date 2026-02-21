import os
import sys
# Add user site-packages to path
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import soundfile as sf
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Configuration
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "Hindi"
TASK = "transcribe"
OUTPUT_DIR = "whisper-small-hi-lora"
TRAIN_CSV = "train.csv"

class WhisperDataset(TorchDataset):
    """Custom PyTorch Dataset that loads audio on-the-fly to avoid memory issues."""
    
    def __init__(self, dataframe, feature_extractor, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        sentence = str(row["sentence"])
        
        # Load audio using soundfile
        speech, sr = sf.read(audio_path)
        
        # Convert to float32 numpy
        speech = np.array(speech, dtype=np.float32)
        
        # Handle stereo -> mono
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        
        # Resample if needed
        if sr != 16000:
            speech_tensor = torch.tensor(speech)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech_tensor).numpy()
        
        # Compute log-Mel input features
        input_features = self.feature_extractor(speech, sampling_rate=16000).input_features[0]
        
        # Encode target text to label ids
        labels = self.tokenizer(sentence).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels,
        }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

if __name__ == "__main__":
    # Check if train.csv exists
    if not os.path.exists(TRAIN_CSV):
        print(f"{TRAIN_CSV} not found. Please run preprocess_dataset.py first.")
        exit(1)

    # Load processors
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    
    # Load CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples from {TRAIN_CSV}")
    
    # Filter out rows with missing audio files
    valid_mask = df["audio_path"].apply(os.path.exists)
    df = df[valid_mask].reset_index(drop=True)
    print(f"Found {len(df)} valid audio files")
    
    # Split into train/test (90/10)
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")
    
    # Create custom datasets (audio loaded on-the-fly, no Arrow memory issues)
    train_dataset = WhisperDataset(train_df, feature_extractor, tokenizer)
    eval_dataset = WhisperDataset(eval_df, feature_extractor, tokenizer)

    # Load Metric
    metric = evaluate.load("wer")

    # Load Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Apply LoRA
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=50,
        max_steps=100, # Short run for demo
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    trainer.train()
    
    # Save
    trainer.save_model(OUTPUT_DIR)
