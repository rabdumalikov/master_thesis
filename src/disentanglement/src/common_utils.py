from transformers.optimization import Adafactor, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration

def create_tokenizer(model_name: str) -> T5Tokenizer:
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print("Finished loading tokenizer")

    return tokenizer

def create_optimizer(model: T5ForConditionalGeneration):
    # print("AdamW optimizer!")
    # return AdamW(model.parameters(), lr=0.0001), None
    return Adafactor(
        model.parameters(),
        lr=0.001,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        ), None