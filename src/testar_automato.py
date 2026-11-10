from automato import AutomatoJSONPilha

pda = AutomatoJSONPilha()

# Teste 1: JSON completo com números, booleanos, null e arrays aninhados
json_teste = '{"id": 101, "ativo": true, "obs": null, "tags": ["ia", "automatos"]}'

valido = pda.validar_sequencia(json_teste) and pda.esta_em_estado_final()
print("JSON Completo é Válido?:", valido) # Deve imprimir True