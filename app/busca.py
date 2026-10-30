"""Etapa de RECUPERAÇÃO do RAG (a parte "R").

1. Quebra cada documento em trechos (um trecho por seção "## ").
2. Usa o algoritmo BM25 para dar uma nota a cada trecho em relação à pergunta.

BM25 foi implementado à mão, sem biblioteca, para deixar claro como a busca
funciona. Em produção daria para trocar por embeddings + banco vetorial.
"""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "base_conhecimento"

STOPWORDS = set(
    "a o as os de da do das dos e em no na nos nas um uma uns umas para por com sem que qual quais "
    "como se ou ao aos e é eu meu minha voce voces fazer faz devo deve tem ter quanto quantas quantos "
    "isso esta este essa esse mais muito".split()
)


@dataclass
class Trecho:
    id: str          # ex.: "tratamento-da-agua#ph-ideal"
    documento: str
    titulo: str
    texto: str


def normalizar(texto: str) -> list[str]:
    texto = "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))
    palavras = re.findall(r"[a-z0-9]+", texto)
    # "radical" simples: tira plural e algumas terminações, para "filtros" achar "filtro"
    return [re.sub(r"(s|es)$", "", p) if len(p) > 4 else p for p in palavras if p not in STOPWORDS]


def _slug(texto: str) -> str:
    return "-".join(normalizar(texto)) or "secao"


def carregar_trechos(pasta: Path = PASTA_PADRAO) -> list[Trecho]:
    trechos = []
    for arquivo in sorted(pasta.glob("*.md")):
        documento = arquivo.stem
        for bloco in re.split(r"^## ", arquivo.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]:
            titulo, _, corpo = bloco.partition("\n")
            trechos.append(Trecho(f"{documento}#{_slug(titulo)}", documento, titulo.strip(), corpo.strip()))
    return trechos


class IndiceBM25:
    def __init__(self, trechos: list[Trecho], k1: float = 1.5, b: float = 0.75):
        self.trechos = trechos
        self.k1, self.b = k1, b
        # o título conta duas vezes: ele resume bem o assunto do trecho
        self.docs = [normalizar(f"{t.titulo} {t.titulo} {t.texto}") for t in trechos]
        self.tamanho_medio = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.frequencias = [Counter(d) for d in self.docs]
        df = Counter(palavra for d in self.docs for palavra in set(d))
        n = len(self.docs)
        self.idf = {p: math.log(1 + (n - q + 0.5) / (q + 0.5)) for p, q in df.items()}

    def buscar(self, pergunta: str, k: int = 3) -> list[tuple[Trecho, float]]:
        termos = normalizar(pergunta)
        notas = []
        for i, freq in enumerate(self.frequencias):
            nota = 0.0
            for termo in termos:
                if termo not in freq:
                    continue
                f = freq[termo]
                nota += self.idf[termo] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.tamanho_medio)
                )
            if nota > 0:
                notas.append((self.trechos[i], round(nota, 3)))
        return sorted(notas, key=lambda x: x[1], reverse=True)[:k]
