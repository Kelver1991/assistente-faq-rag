import pytest
from fastapi.testclient import TestClient

from app import assistente
from app.busca import normalizar
from app.main import INDICE, app

cliente = TestClient(app)


@pytest.fixture(autouse=True)
def sem_chave(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def perguntar(texto):
    return cliente.post("/perguntas", json={"pergunta": texto}).json()


@pytest.mark.parametrize("pergunta, esperado", [
    ("Qual o pH ideal da piscina?", "tratamento-da-agua#ph-ideal"),
    ("Minha água está verde, o que faço?", "tratamento-da-agua#agua-verde"),
    ("Quando fazer a retrolavagem do filtro?", "equipamentos#retrolavagem-filtro"),
    ("Quantas horas devo deixar o filtro ligado?", "equipamentos#tempo-filtragem"),
    ("A garantia cobre a instalação?", "atendimento#garantia"),
    ("Como abro um chamado de assistência?", "atendimento#abrir-chamado"),
])
def test_busca_encontra_o_trecho_certo(pergunta, esperado):
    assert INDICE.buscar(pergunta, k=1)[0][0].id == esperado


def test_resposta_demo_cita_a_fonte():
    r = perguntar("Qual o pH ideal?")
    assert r["modo"] == "demo"
    assert "7,2 e 7,6" in r["resposta"]
    assert r["fontes"][0]["id"] == "tratamento-da-agua#ph-ideal"


def test_pergunta_fora_da_base_nao_inventa():
    r = perguntar("Qual a capital da França?")
    assert r["resposta"] == assistente.NAO_SEI and r["fontes"] == []


def test_ia_recebe_apenas_os_trechos(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "teste")
    recebido = {}

    def falso(sistema, mensagem, max_tokens=500):
        recebido["m"] = mensagem
        return "Entre 7,2 e 7,6. [tratamento-da-agua#ph-ideal]"

    monkeypatch.setattr(assistente.llm, "gerar_texto", falso)
    r = perguntar("Qual o pH ideal?")
    assert r["modo"] == "ia"
    assert '<trecho id="tratamento-da-agua#ph-ideal"' in recebido["m"]
    assert "Pergunta do cliente: Qual o pH ideal?" in recebido["m"]


def test_normalizacao_ignora_acento_e_plural():
    assert normalizar("Filtros") == normalizar("filtro")
    assert normalizar("água") == ["agua"]


def test_lista_documentos():
    ids = [d["id"] for d in cliente.get("/documentos").json()]
    assert len(ids) == len(set(ids)) == 14
