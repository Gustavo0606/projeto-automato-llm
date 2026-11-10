"""
automato.py
Autômato com Pilha (PDA) para validação sintática de JSON em tempo de decoding.
"""

from collections import deque

class AutomatoJSONPilha:
    def __init__(self):
        self.pilha = deque()
        self.estado_atual = "ESPERANDO_ABERTURA"
        self.em_string = False
        self.escape = False
        self.valido = True

    def reset(self):
        """Reinicia o autômato para o estado inicial."""
        self.pilha.clear()
        self.estado_atual = "ESPERANDO_ABERTURA"
        self.em_string = False
        self.escape = False
        self.valido = True

    def clone(self):
        """Cria uma cópia leve e rápida do autômato para simulação de tokens."""
        novo = AutomatoJSONPilha()
        novo.pilha = self.pilha.copy()
        novo.estado_atual = self.estado_atual
        novo.em_string = self.em_string
        novo.escape = self.escape
        novo.valido = self.valido
        return novo

    def processar_caractere(self, char: str) -> bool:
        """
        Lê um caractere e executa a transição de estado e pilha.
        Retorna True se for válido, False se violar a sintaxe.
        """
        if not self.valido:
            return False

        if self.em_string:
            if self.escape:
                self.escape = False
                return True
            if char == '\\':
                self.escape = True
                return True
            if char == '"':
                self.em_string = False
                if self.estado_atual == "LENDO_CHAVE":
                    self.estado_atual = "APOS_CHAVE"
                else:
                    self.estado_atual = "DEPOIS_VALOR"
                return True
            return True 

        if char in (' ', '\n', '\t', '\r'):
            return True

        if self.estado_atual == "ESPERANDO_ABERTURA":
            if char == '{':
                self.pilha.append('}')
                self.estado_atual = "ESPERANDO_CHAVE"
                return True
            elif char == '[':
                self.pilha.append(']')
                self.estado_atual = "ESPERANDO_VALOR"
                return True
            return False

        elif self.estado_atual == "FINALIZADO":
            return False

        elif self.estado_atual == "ESPERANDO_CHAVE":
            if char == '"':
                self.em_string = True
                self.estado_atual = "LENDO_CHAVE"
                return True
            elif char == '}' and self.pilha and self.pilha[-1] == '}':
                self.pilha.pop()
                if len(self.pilha) == 0:
                    self.estado_atual = "FINALIZADO"
                else:
                    self.estado_atual = "DEPOIS_VALOR"
                return True
            return False

        elif self.estado_atual == "APOS_CHAVE":
            if char == ':':
                self.estado_atual = "ESPERANDO_VALOR"
                return True
            return False

        elif self.estado_atual == "ESPERANDO_VALOR":
            if char == '"':
                self.em_string = True
                return True
            elif char == '{':
                self.pilha.append('}')
                self.estado_atual = "ESPERANDO_CHAVE"
                return True
            elif char == '[':
                self.pilha.append(']')
                self.estado_atual = "ESPERANDO_VALOR"
                return True
            elif char.isdigit() or char == '-':
                self.estado_atual = "LENDO_NUMERO"
                return True
            elif char in ('t', 'f', 'n'): 
                self.estado_atual = "LENDO_LITERAL"
                return True
            elif char == ']' and self.pilha and self.pilha[-1] == ']':
                self.pilha.pop()
                if len(self.pilha) == 0:
                    self.estado_atual = "FINALIZADO"
                else:
                    self.estado_atual = "DEPOIS_VALOR"
                return True
            return False

        elif self.estado_atual == "LENDO_NUMERO":
            if char.isdigit() or char == '.':
                return True
            self.estado_atual = "DEPOIS_VALOR"
            return self.processar_caractere(char)

        elif self.estado_atual == "LENDO_LITERAL":
            if char.isalpha():
                return True
            self.estado_atual = "DEPOIS_VALOR"
            return self.processar_caractere(char)

        elif self.estado_atual == "DEPOIS_VALOR":
            if char == ',':
                if self.pilha and self.pilha[-1] == '}':
                    self.estado_atual = "ESPERANDO_CHAVE"
                else:
                    self.estado_atual = "ESPERANDO_VALOR"
                return True
            elif char in ('}', ']'):
                if not self.pilha or self.pilha[-1] != char:
                    self.valido = False
                    return False
                self.pilha.pop()
                if len(self.pilha) == 0:
                    self.estado_atual = "FINALIZADO"
                else:
                    self.estado_atual = "DEPOIS_VALOR"
                return True
            return False

        return False

    def validar_sequencia(self, texto: str) -> bool:
        """Processa uma string inteira caractere a caractere."""
        for c in texto:
            if not self.processar_caractere(c):
                return False
        return self.valido

    def esta_em_estado_final(self) -> bool:
        """O JSON só é aceito se estiver no estado FINALIZADO ou com pilha vazia fora de strings."""
        return (len(self.pilha) == 0 or self.estado_atual == "FINALIZADO") and self.valido and not self.em_string