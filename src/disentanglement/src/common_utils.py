from transformers.optimization import Adafactor, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration

def create_tokenizer(model_name: str) -> T5Tokenizer:
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print("Finished loading tokenizer")

    return tokenizer

def create_optimizer(model: T5ForConditionalGeneration, learning_rate: float, type_of_optimizer: str):
    print(f"Optimizer {type_of_optimizer}")

    if type_of_optimizer == 'adamw':
        return AdamW(model.parameters(), lr=learning_rate), None
    else:
        return Adafactor(
            model.parameters(),
            lr=learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            ), None