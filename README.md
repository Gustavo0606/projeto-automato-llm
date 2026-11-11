# Decodificação Restrita Sintaticamente em LLMs via Autômato de Pilha Determinístico (PDA)

Este repositório contém a implementação empírica e a suíte de benchmarks para validação de **Hard Constraints (Restrições Rígidas)** sintáticas durante a fase de decodificação de modelos de linguagem de pequeno porte (*Small Language Models* — SLMs).

O projeto implementa um **Processador de Logits baseado em Autômato de Pilha Determinístico (Pushdown Automaton - PDA Logits Processor)** para a gramática **JSON (RFC 8259)**, garantindo 100% de conformidade sintática nativa no modelo `Qwen/Qwen2.5-0.5B-Instruct`.

---

## 📌 Sumário
- [Visão Geral e Problema](#visão-geral-e-problema)
- [Fundamentação Teórica e Arquitetura](#fundamentação-teórica-e-arquitetura)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Metodologia Experimental (3 Cenários)](#metodologia-experimental-3-cenários)
- [Resultados do Benchmark](#resultados-do-benchmark)
- [Principais Achados Científicos](#principais-achados-científicos)
- [Como Executar o Projeto](#como-executar-o-projeto)

---

## 🎯 Visão Geral e Problema

O uso de Grandes Modelos de Linguagem (LLMs) como geradores de dados estruturados para integração via APIs de backend enfrenta um problema central de confiabilidade: a ocorrência de alucinações sintáticas e violações de formato.

### Limitações das Abordagens Tradicionais (*Soft Constraints*)
1. **Ruído Conversacional:** Modelos alinhados por instrução tendem a prefixar e sufixar JSONs com saudações e explicações ("*Aqui está o seu JSON:*").
2. **Poluição por Markdown:** Viés acentuado de *Supervised Fine-Tuning* (SFT) que força a inclusão de delimitadores de código (` ```json ... ``` `).
3. **Erros de Fechamento e Escapamento:** Esquecimento de aspas de fechamento, vírgulas sobressalentes (*trailing commas*) ou falta de balanceamento de chaves `{}` e colchetes `[]`.

---

## 🧬 Fundamentação Teórica e Arquitetura

Como a linguagem JSON é uma **Linguagem Livre de Contexto (Context-Free Grammar — CFG)** devido à sua natureza recursiva de aninhamento de objetos e arrays, ela não pode ser validada por autômatos finitos simples (Expressões Regulares). É necessária a memória de uma estrutura de pilha.

### 1. Autômato de Pilha Determinístico (`src/automato.py`)
O módulo `AutomatoJSONPilha` rastreia o estado da geração a cada token, empilhando (`PUSH`) e desempilhando (`POP`) símbolos estruturais:
* **Escopos de Pilha:** `{` para objetos, `[` para arrays, `"` para strings.
* **Transição de Estados:** Garante a sequência gramatical estrita `Chave -> Dois-pontos -> Valor -> Vírgula/Fechamento`.

### 2. Máscara de Logits na Inferência (`src/logit_processor.py`)
A classe `JSONLogitsProcessor` intercepta o vetor de distribuição de probabilidade desnormalizada (*logits*) $\mathbf{z}_t \in \mathbb{R}^{|V|}$ gerado pelo modelo no passo de tempo $t$:

$$\mathbf{z}_t^{\text{filtrado}}[i] = \begin{cases} \mathbf{z}_t[i], & \text{se o token } i \text{ for sintaticamente válido no estado atual do PDA} \\ -\infty, & \text{caso contrário} \end{cases}$$

Ao aplicar a função *Softmax* sobre o vetor filtrado, a probabilidade de selecionar qualquer token que viole a gramática JSON é reduzida a exatamente zero ($P(v_i) = 0$).

---

## 📂 Estrutura do Repositório

```text
projeto-automato-llm/
├── experimentos/
│   └── benchmark_metricas.py    # Script principal de execução da bateria quantitativa
├── resultados/
│   ├── relatorio_benchmark_3_cenarios.csv   # Métricas quantitativas (Validade e Latência)
│   └── respostas_detalhadas.json             # Log qualitativo dos textos brutos gerados
├── src/
│   ├── automato.py              # Implementação do Autômato de Pilha
│   └── logit_processor.py       # Integração com a API LogitsProcessor do PyTorch/Transformers
└── README.md
```

---

## 🔬 Metodologia Experimental (3 Cenários)

O benchmark submete $N=10$ prompts representativos de estruturas de dados diversificadas (aninhamento, arrays, tipos primitivos, nulos e booleanos) a três cenários de controle:

1. **Cenário 1 — Baseline Ingênuo (Sem Restrição):**  
   Prompt direto de usuário sem qualquer instrução de sistema ou formatação.
2. **Cenário 2 — Baseline Refinado (*Soft Constraint* / Prompt Engineering):**  
   Uso de *System Prompt* rigoroso exigindo apenas o JSON puro e proibindo expressamente o uso de Markdown ou explicações.
3. **Cenário 3 — PDA Constraint (*Hard Constraint* / Autômato):**  
   Execução do prompt simples acoplado ao `JSONLogitsProcessor` na camada de decodificação.

---

## 📊 Resultados do Benchmark

### Tabela Comparativa de Desempenho

| Cenário | Tipo de Restrição | Validade Estrita (RFC 8259) | Validade Pós-Processada* | Tempo Médio por Saída |
| :--- | :--- | :---: | :---: | :---: |
| **1. Baseline Ingênuo** | Nenhuma | **0.0%** | **0.0%** | ~17.43 s |
| **2. Baseline Refinado** | *Soft Constraint* (Prompt Eng.) | **0.0%** | **100.0%** | ~6.01 s |
| **3. PDA Logits Processor** | *Hard Constraint* (Autômato) | **100.0%** | **100.0%** | ~35.68 s |

*\*Validade Pós-Processada indica a taxa de sucesso caso um sanitizador regex externo remova as marcas de bloco Markdown (` ```json `).*

---

## 💡 Principais Achados Científicos

1. **Incapacidade de Restrição Negativa em SLMs (Viés de SFT):**  
   Em modelos de 0.5B parâmetros (`Qwen2.5-0.5B-Instruct`), instruções negativas no *System Prompt* (ex: *"NÃO inclua markdown"*) são sistematicamente ignoradas. O modelo produziu blocos ` ```json ` em 100% das amostras no Cenário 2 devido ao forte viés de alinhamento prévio.

2. **Garantia Sintática Estrita Nativa:**  
   O **PDA Logits Processor** foi o único método capaz de entregar 100% de conformidade com o padrão estrito **RFC 8259**, eliminando a necessidade de scripts de sanitização ou *parsers* corretivos no lado do cliente.

3. **Eficiência de Janela de Contexto:**  
   Enquanto o *Prompt Engineering* tradicional consome tokens de entrada adicionais para instruir o formato (aumentando o custo e latência de *prefill*), o autômato garante conformidade sintática com overhead zero de tokens de contexto.

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
* Python 3.10+
* Virtualenv ativo (`venv`)

### 1. Instalação das Dependências
```powershell
pip install torch transformers
```

### 2. Execução da Bateria de Benchmarks
```powershell
python experimentos/benchmark_metricas.py
```

Os resultados quantitativos serão exportados para `resultados/relatorio_benchmark_3_cenarios.csv` e os logs de resposta detalhados para `resultados/respostas_detalhadas.json`.