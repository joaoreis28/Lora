import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor


def get_clip_lora_model(
    model_name="openai/clip-vit-base-patch32",
    r=16,
    lora_alpha=32,       
                         
                         
    lora_dropout=0.1
):
   

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        
    )

    
    peft_model = get_peft_model(model, config)

    
    peft_model.print_trainable_parameters()
    return peft_model, processor