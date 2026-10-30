"""Etapa de GERAÇÃO do RAG (a parte "G").

O modelo só pode responder com base nos trechos encontrados pela busca.
Isso reduz alucinação e permite mostrar a fonte de cada resposta.
"""
from . import llm
from .busca import IndiceBM25, Trecho

NOTA_MINIMA = 1.0  # abaixo disso consideramos que a base não tem a resposta

NAO_SEI = "Não encontrei essa informação na base de conhecimento. Vou encaminhar para um consultor."

PROMPT_SISTEMA = """Você é o assistente de atendimento de uma loja de piscinas.
Responda à pergunta usando SOMENTE os trechos fornecidos entre <trechos>.
- Se os trechos não tiverem a resposta, responda exatamente: "{nao_sei}"
- Cite a fonte no fim de cada informação usando o id entre colchetes, ex.: [equipamentos#retrolavagem-filtro].
- Resposta curta (até 5 frases), em português do Brasil, linguagem simples.
- Não dê orientação de segurança elétrica ou química além do que está nos trechos.
""".format(nao_sei=NAO_SEI)


def _montar_contexto(encontrados: list[tuple[Trecho, float]]) -> str:
    partes = [f'<trecho id="{t.id}" titulo="{t.titulo}">\n{t.texto}\n</trecho>' for t, _ in encontrados]
    return "<trechos>\n" + "\n".join(partes) + "\n</trechos>"


def responder(indice: IndiceBM25, pergunta: str) -> dict:
    encontrados = [(t, n) for t, n in indice.buscar(pergunta, k=3) if n >= NOTA_MINIMA]
    fontes = [{"id": t.id, "titulo": t.titulo, "nota": n} for t, n in encontrados]

    if not encontrados:
        return {"modo": "ia" if llm.ia_disponivel() else "demo", "resposta": NAO_SEI, "fontes": []}

    if not llm.ia_disponivel():
        melhor = encontrados[0][0]
        return {"modo": "demo", "resposta": f"{melhor.texto} [{melhor.id}]", "fontes": fontes}

    mensagem = f"{_montar_contexto(encontrados)}\n\nPergunta do cliente: {pergunta}"
    resposta = llm.gerar_texto(PROMPT_SISTEMA, mensagem, max_tokens=500)
    return {"modo": "ia", "resposta": resposta, "fontes": fontes}
