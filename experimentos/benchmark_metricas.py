"""
benchmark_metricas.py
Bateria quantitativa de 3 cenários na CPU:
 1. Baseline Ingênuo (Sem instrução sintática)
 2. Baseline Refinado (Engenharia de Prompt / Soft Constraint)
 3. PDA Filter (Restrição determinística via Autômato / Hard Constraint)
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

SYSTEM_PROMPT_RIGOROSO = (
    "Você é um gerador de dados no formato JSON estrito. "
    "Responda APENAS com o objeto ou array JSON válido correspondente. "
    "NÃO inclua formatação Markdown (como ```json), explicações, saudações ou texto adicional."
)

def avaliar_resposta(texto: str) -> bool:
    """Valida se a string gerada é um JSON estritamente válido segundo a RFC 8259."""
    try:
        json.loads(texto.strip())
        return True
    except Exception:
        return False

def main():
    device = "cpu"
    dtype = torch.float32
    max_tokens = 200 
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"=== BENCHMARK DE 3 CENÁRIOS ({model_id}) ===")
    print(f"Dispositivo: {device.upper()} | Limite de Tokens: {max_tokens}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    os.makedirs("resultados", exist_ok=True)
    arquivo_csv = "resultados/relatorio_benchmark_3_cenarios.csv"
    arquivo_json = "resultados/respostas_detalhadas.json"
    resultados = []

    total = len(DATASET_PROMPTS)
    print(f"Executando bateria com {total} prompts em 3 cenários...\n")

    for idx, prompt in enumerate(DATASET_PROMPTS, 1):
        porcentagem = (idx / total) * 100
        print(f"[{idx}/{total} | {porcentagem:.0f}%] Prompt: \"{prompt[:45]}...\"")

        # -------------------------------------------------------------
        # CENÁRIO 1: BASELINE INGÊNUO (Prompt Simples)
        # -------------------------------------------------------------
        print("  ├── [1/3] Baseline Ingênuo...", end="", flush=True)
        msg_ingenuo = [{"role": "user", "content": prompt}]
        fmt_ingenuo = tokenizer.apply_chat_template(msg_ingenuo, tokenize=False, add_generation_prompt=True)
        inp_ingenuo = tokenizer(fmt_ingenuo, return_tensors="pt").to(device)
        len_ingenuo = inp_ingenuo["input_ids"].shape[1]

        t0 = time.time()
        out_ingenuo = model.generate(
            **inp_ingenuo,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        t_ingenuo = round(time.time() - t0, 2)
        txt_ingenuo = tokenizer.decode(out_ingenuo[0][len_ingenuo:], skip_special_tokens=True)
        val_ingenuo = avaliar_resposta(txt_ingenuo)
        print(f" {t_ingenuo}s (Válido: {val_ingenuo})")

        # -------------------------------------------------------------
        # CENÁRIO 2: BASELINE REFINADO (System Prompt / Soft Constraint)
        # -------------------------------------------------------------
        print("  ├── [2/3] Baseline Refinado (Prompt Eng.)...", end="", flush=True)
        msg_refinado = [
            {"role": "system", "content": SYSTEM_PROMPT_RIGOROSO},
            {"role": "user", "content": prompt}
        ]
        fmt_refinado = tokenizer.apply_chat_template(msg_refinado, tokenize=False, add_generation_prompt=True)
        inp_refinado = tokenizer(fmt_refinado, return_tensors="pt").to(device)
        len_refinado = inp_refinado["input_ids"].shape[1]

        t0 = time.time()
        out_refinado = model.generate(
            **inp_refinado,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        t_refinado = round(time.time() - t0, 2)
        txt_refinado = tokenizer.decode(out_refinado[0][len_refinado:], skip_special_tokens=True)
        val_refinado = avaliar_resposta(txt_refinado)
        print(f" {t_refinado}s (Válido: {val_refinado})")

        # -------------------------------------------------------------
        # CENÁRIO 3: PDA CONSTRAINT (Filtro Determinístico / Hard Constraint)
        # -------------------------------------------------------------
        print("  └── [3/3] PDA Constraint...", end="", flush=True)
        automato = AutomatoJSONPilha()
        processor = JSONLogitsProcessor(tokenizer, automato, len_ingenuo)
        logits_processors = LogitsProcessorList([processor])

        t0 = time.time()
        out_pda = model.generate(
            **inp_ingenuo,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            logits_processor=logits_processors,
            pad_token_id=tokenizer.eos_token_id
        )
        t_pda = round(time.time() - t0, 2)
        txt_pda = tokenizer.decode(out_pda[0][len_ingenuo:], skip_special_tokens=True)
        val_pda = avaliar_resposta(txt_pda)
        print(f" {t_pda}s (Válido: {val_pda})")

        resultados.append({
            "id": idx,
            "prompt": prompt,
            "base_ingenuo_valido": val_ingenuo,
            "base_ingenuo_tempo_s": t_ingenuo,
            "base_ingenuo_texto": txt_ingenuo,
            "base_refinado_valido": val_refinado,
            "base_refinado_tempo_s": t_refinado,
            "base_refinado_texto": txt_refinado,
            "pda_valido": val_pda,
            "pda_tempo_s": t_pda,
            "pda_texto": txt_pda
        })

    # -------------------------------------------------------------
    # SALVAR RELATÓRIO EM CSV (Ignorando os campos de texto no CSV)
    # -------------------------------------------------------------
    campos_csv = [
        "id", "prompt", 
        "base_ingenuo_valido", "base_ingenuo_tempo_s",
        "base_refinado_valido", "base_refinado_tempo_s",
        "pda_valido", "pda_tempo_s"
    ]
    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore" impede a exceção ao descartar as colunas de texto do CSV
        writer = csv.DictWriter(f, fieldnames=campos_csv, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resultados)

    # -------------------------------------------------------------
    # SALVAR DETALHES DAS RESPOSTAS EM JSON
    # -------------------------------------------------------------
    with open(arquivo_json, mode="w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------
    # RESUMO ESTATÍSTICO
    # -------------------------------------------------------------
    acc_ingenuo = (sum(1 for r in resultados if r["base_ingenuo_valido"]) / total) * 100
    acc_refinado = (sum(1 for r in resultados if r["base_refinado_valido"]) / total) * 100
    acc_pda = (sum(1 for r in resultados if r["pda_valido"]) / total) * 100

    med_t_ingenuo = sum(r["base_ingenuo_tempo_s"] for r in resultados) / total
    med_t_refinado = sum(r["base_refinado_tempo_s"] for r in resultados) / total
    med_t_pda = sum(r["pda_tempo_s"] for r in resultados) / total

    print("\n" + "="*65)
    print("RESUMO COMPARAÇÕES DE DESEMPENHO (3 CENÁRIOS)")
    print("="*65)
    print(f"Total de Amostras:                    {total}")
    print(f"Acurácia Baseline Ingênuo:            {acc_ingenuo:.1f}% | Tempo Médio: {med_t_ingenuo:.2f}s")
    print(f"Acurácia Baseline Refinado (Prompt):  {acc_refinado:.1f}% | Tempo Médio: {med_t_refinado:.2f}s")
    print(f"Acurácia PDA Filter (Autômato):       {acc_pda:.1f}% | Tempo Médio: {med_t_pda:.2f}s")
    print("="*65)
    print(f"Relatório CSV salvo em:  {arquivo_csv}")
    print(f"Log detalhado salvo em:  {arquivo_json}")

if __name__ == "__main__":
    main()