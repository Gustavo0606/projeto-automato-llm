"""
rodar_comparativo.py
Script de teste comparativo: geração de JSON com e sem LogitsProcessor determinístico.
"""

import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from automato import AutomatoJSONPilha
from logit_processor import JSONLogitsProcessor

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"--- Carregando o modelo {model_id} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32  
    )

    prompt = "Gere um JSON válido com a ficha de um personagem contendo nome, idade (numero), classe e um array de equipamentos (strings)."
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]

    print("\n==================================================")
    print("PROMPT:")
    print(prompt)
    print("==================================================\n")

    # -------------------------------------------------------------
    # RODADA 1: GERAÇÃO SEM FILTRO (BASELINE / SEM CONSTRAINTS)
    # -------------------------------------------------------------
    print("--- 1. Executando Geração SEM Filtro (Baseline) ---")
    output_sem_filtro = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    texto_sem_filtro = tokenizer.decode(output_sem_filtro[0][prompt_length:], skip_special_tokens=True)
    print("Resposta Gerada:\n", texto_sem_filtro)
    
    # Validação via Python nativo
    try:
        json.loads(texto_sem_filtro)
        print(">> Status SEM filtro: JSON 100% VÁLIDO")
    except Exception as e:
        print(f">> Status SEM filtro: FALHOU SINTATICAMENTE ({e})")

    # -------------------------------------------------------------
    # RODADA 2: GERAÇÃO COM FILTRO DETERMINÍSTICO (PDA + TRANSDUTOR)
    # -------------------------------------------------------------
    print("\n--- 2. Executando Geração COM Filtro (PDA LogitsProcessor) ---")
    
    # Instancia o autômato e o processador de logits
    automato = AutomatoJSONPilha()
    json_processor = JSONLogitsProcessor(tokenizer, automato, prompt_length)
    logits_processors = LogitsProcessorList([json_processor])

    output_com_filtro = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        logits_processor=logits_processors,
        pad_token_id=tokenizer.eos_token_id
    )
    
    texto_com_filtro = tokenizer.decode(output_com_filtro[0][prompt_length:], skip_special_tokens=True)
    print("Resposta Gerada:\n", texto_com_filtro)

    # Validação via Python nativo
    try:
        json.loads(texto_com_filtro)
        print(">> Status COM filtro: JSON 100% VÁLIDO")
    except Exception as e:
        print(f">> Status COM filtro: FALHOU SINTATICAMENTE ({e})")

if __name__ == "__main__":
    main()