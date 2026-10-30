from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import assistente, llm
from .busca import IndiceBM25, carregar_trechos

app = FastAPI(
    title="Assistente de Dúvidas com RAG",
    description="Responde dúvidas de clientes de piscinas usando busca em documentos + IA, sempre citando a fonte.",
    version="1.0.0",
)

# O índice é montado uma vez, quando a API sobe
TRECHOS = carregar_trechos()
INDICE = IndiceBM25(TRECHOS)


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)


class Fonte(BaseModel):
    id: str
    titulo: str
    nota: float


class Resposta(BaseModel):
    modo: str
    resposta: str
    fontes: list[Fonte]


@app.get("/saude")
def saude():
    return {"status": "ok", "modo": "ia" if llm.ia_disponivel() else "demo", "trechos_indexados": len(TRECHOS)}


@app.get("/documentos")
def documentos():
    return [{"id": t.id, "titulo": t.titulo} for t in TRECHOS]


@app.post("/perguntas", response_model=Resposta)
def perguntar(p: Pergunta):
    try:
        return assistente.responder(INDICE, p.pergunta)
    except Exception as erro:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar a IA: {type(erro).__name__}")
