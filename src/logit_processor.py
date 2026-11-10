"""
logit_processor.py
Transdutor Finito integrado ao PyTorch / Hugging Face via LogitsProcessor.
"""

import torch
from transformers import LogitsProcessor
from automato import AutomatoJSONPilha

class JSONLogitsProcessor(LogitsProcessor):
    """
    Filtro determinístico que zera a probabilidade (-inf) de tokens invalidados pelo Autômato.
    """
    def __init__(self, tokenizer, automato_base: AutomatoJSONPilha, prompt_length: int):
        self.tokenizer = tokenizer
        self.automato = automato_base
        self.prompt_length = prompt_length
        self.ultimo_tamanho_seq = prompt_length

        print("--- Pré-processando vocabulário do modelo ---")
        self.vocab_size = len(tokenizer)
        self.vocab_strings = [
            tokenizer.decode([i], skip_special_tokens=False) for i in range(self.vocab_size)
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Intercepta o vetor de logits a cada step de geração.
        """
        tamanho_atual = input_ids.shape[1]

        if tamanho_atual > self.ultimo_tamanho_seq:
            ultimo_token_id = input_ids[0, -1].item()
            texto_ultimo_token = self.vocab_strings[ultimo_token_id]
            self.automato.validar_sequencia(texto_ultimo_token)
            self.ultimo_tamanho_seq = tamanho_atual

        eos_token_id = self.tokenizer.eos_token_id

        for token_id in range(self.vocab_size):
            if token_id == eos_token_id:
                if not self.automato.esta_em_estado_final():
                    scores[0, token_id] = float("-inf")
                continue

            token_str = self.vocab_strings[token_id]
            
            automato_simulado = self.automato.clone()
            
            if not automato_simulado.validar_sequencia(token_str):
                scores[0, token_id] = float("-inf")

        return scores