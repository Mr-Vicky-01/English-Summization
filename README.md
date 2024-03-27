# BART Large CNN Text Summarization Model

This model is based on the Facebook BART (Bidirectional and Auto-Regressive Transformers) architecture, specifically the large variant fine-tuned for text summarization tasks. BART is a sequence-to-sequence model introduced by Facebook AI, capable of handling various natural language processing tasks, including summarization.
![image](https://github.com/Mr-Vicky-01/English-Summization/assets/143078285/026ef20a-3976-4563-9cea-39ff246e5c4a)

## Model Details:

- **Architecture**: BART Large CNN
- **Pre-trained model**: BART Large
- **Fine-tuned for**: Text Summarization
- **Fine-tuning dataset**: [xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)

## Usage:

### Installation:

You can install the necessary libraries using pip:
```bash
pip install transformers
```
### Inferecnce
provided a simple snippet of how to use this model for the task of paragraph summarization in PyTorch.
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```
```python
tokenizer = AutoTokenizer.from_pretrained("Mr-Vicky-01/conversational_sumarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Mr-Vicky-01/conversational_sumarization")

def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

text_to_summarize = """Now, there is no doubt that one of the most important aspects of any Pixel phone is its camera.
And there might be good news for all camera lovers. Rumours have suggested that the Pixel 9 could come with a telephoto lens,
improving its photography capabilities even further. Google will likely continue to focus on using AI to enhance its camera performance,
in order to make sure that Pixel phones remain top contenders in the world of mobile photography."""
summary = generate_summary(text_to_summarize)
print(summary)
```

```Example
Google is rumoured to be about to unveil its next-generation Pixel smartphone,
the Google Pixel 9,which is expected to come with a telephoto lens and an artificial intelligence (AI)
system to improve its camera capabilities, as well as improve the quality of its images.
```

### Training Parameters
```python
num_train_epochs=1,
warmup_steps = 500,
per_device_train_batch_size=4,
per_device_eval_batch_size=4,
weight_decay = 0.01,
gradient_accumulation_steps=16
```

## Demo snap
![image](https://github.com/Mr-Vicky-01/tamil_summarization/assets/143078285/6e8e0a18-ad4f-499b-ad38-a12bb845c0a2)

## Model 
1. [Model link](https://huggingface.co/Mr-Vicky-01/conversational_sumarization)
2. [try this model](https://huggingface.co/spaces/Mr-Vicky-01/Summarization)
