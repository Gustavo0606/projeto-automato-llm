"""
benchmark_metricas.py
Executa uma bateria de testes quantitativos comparando a geração Baseline vs PDA Constraint na CPU.
Exibe progresso em porcentagem e sub-etapas em tempo real no terminal.
"""

import sys
import os
import json
import time
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Garante acesso aos módulos na pasta src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from automato import AutomatoJSONPilha
from logit_processor import JSONLogitsProcessor

# Bateria de 10 prompts com diferentes estruturas e complexidades sintáticas
DATASET_PROMPTS = [
    "Gere um JSON com nome e idade de uma pessoa.",
    "Gere um JSON representando um produto com nome, preco (numero) e disponivel (booleano).",
    "Gere um JSON com uma lista de 3 cidades (strings).",
    "Gere um JSON de uma configuracao de sistema contendo host, porta (numero), debug (booleano) e banco (null).",
    "Gere um JSON com a ficha de um livro contendo titulo, autor e um array de capitulos.",
    "Gere um JSON com o carrinho de compras contendo um array de itens, onde cada item tem produto e quantidade.",
    "Gere um JSON com o resultado de um exame contendo paciente, aprovado (true/false) e notas (array de numeros).",
    "Gere um JSON contendo uma lista de tarefas com id, descricao e concluida.",
    "Gere um JSON de um pedido contendo id_pedido, cliente e endereco aninhado (rua, numero, cep).",
    "Gere um JSON com estatisticas de servidor contendo cpu_usage, memory_free e status_ok."
]

def avaliar_resposta(texto: str) -> bool:
    """Valida se a string gerada é um JSON estritamente válido segundo a RFC 8259."""
    try:
        json.loads(texto.strip())
        return True
    except Exception:
        return False

def main():
    # Execução limpa em CPU
    device = "cpu"
    dtype = torch.float32
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"=== INICIANDO BENCHMARK QUANTITATIVO ({model_id}) ===")
    print(f"Dispositivo de execução: {device.upper()} | Precisão: {dtype}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)

    # Garante a criação do diretório de saída
    os.makedirs("resultados", exist_ok=True)
    arquivo_csv = "resultados/relatorio_benchmark.csv"
    resultados = []

    total = len(DATASET_PROMPTS)
    print(f"Executando bateria com {total} prompts...\n")

    for idx, prompt in enumerate(DATASET_PROMPTS, 1):
        porcentagem = (idx / total) * 100
        print(f"[{idx}/{total} | {porcentagem:.0f}%] Processando Prompt: \"{prompt[:45]}...\"")

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        # -------------------------------------------------------------
        # 1. TESTE BASELINE (SEM FILTRO)
        # -------------------------------------------------------------
        print("  ├── [1/2] Executando Baseline (Sem filtro)...", end="", flush=True)
        t0 = time.time()
        out_base = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        tempo_base = round(time.time() - t0, 2)
        txt_base = tokenizer.decode(out_base[0][prompt_length:], skip_special_tokens=True)
        valido_base = avaliar_resposta(txt_base)
        print(f" Concluído em {tempo_base}s! (Válido: {valido_base})")

        # -------------------------------------------------------------
        # 2. TESTE PDA CONSTRAINT (COM FILTRO)
        # -------------------------------------------------------------
        print("  └── [2/2] Executando PDA Restrito (Com filtro)...", end="", flush=True)
        automato = AutomatoJSONPilha()
        processor = JSONLogitsProcessor(tokenizer, automato, prompt_length)
        logits_processors = LogitsProcessorList([processor])

        t0 = time.time()
        out_pda = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            logits_processor=logits_processors,
            pad_token_id=tokenizer.eos_token_id
        )
        tempo_pda = round(time.time() - t0, 2)
        txt_pda = tokenizer.decode(out_pda[0][prompt_length:], skip_special_tokens=True)
        valido_pda = avaliar_resposta(txt_pda)
        print(f" Concluído em {tempo_pda}s! (Válido: {valido_pda})")

        resultados.append({
            "id": idx,
            "prompt": prompt,
            "base_valido": valido_base,
            "base_tempo_s": tempo_base,
            "pda_valido": valido_pda,
            "pda_tempo_s": tempo_pda
        })

    # -------------------------------------------------------------
    # EXPEDIÇÃO DO RELATÓRIO EM CSV
    # -------------------------------------------------------------
    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "base_valido", "base_tempo_s", "pda_valido", "pda_tempo_s"])
        writer.writeheader()
        writer.writerows(resultados)

    acc_base = (sum(1 for r in resultados if r["base_valido"]) / total) * 100
    acc_pda = (sum(1 for r in resultados if r["pda_valido"]) / total) * 100
    media_tempo_base = sum(r["base_tempo_s"] for r in resultados) / total
    media_tempo_pda = sum(r["pda_tempo_s"] for r in resultados) / total

    print("\n" + "="*60)
    print("RESUMO DAS MÉTRICAS DE DESEMPENHO")
    print("="*60)
    print(f"Total de Amostras:              {total}")
    print(f"Acurácia Sintática Baseline:     {acc_base:.1f}%")
    print(f"Acurácia Sintática PDA Filter:  {acc_pda:.1f}%")
    print(f"Tempo Médio por Saída Baseline: {media_tempo_base:.2f} s")
    print(f"Tempo Médio por Saída PDA:      {media_tempo_pda:.2f} s")
    print("="*60)
    print(f"Relatório completo salvo em: {arquivo_csv}")

if __name__ == "__main__":
    main()