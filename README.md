# Assistente de Dúvidas com RAG

API em **FastAPI** que responde dúvidas de clientes de uma loja de piscinas usando **RAG (Retrieval-Augmented Generation)**: primeiro busca os trechos certos nos documentos da loja, depois o **LLM (Claude)** responde usando só esses trechos e **cita a fonte**.

## O problema

Clientes perguntam sempre as mesmas coisas: pH ideal, água verde, quanto tempo deixar o filtro ligado, como acionar a garantia. Um chatbot comum pode inventar respostas. Aqui a resposta precisa vir dos documentos da empresa, e quando a informação não existe, o assistente diz que não sabe e encaminha para um consultor.

## Como funciona

```
pergunta do cliente
      │
      ▼
[R] busca.py: BM25 dá nota para cada trecho dos documentos
      │
      ├── nenhum trecho com nota mínima? ──► "Não encontrei, vou encaminhar a um consultor"
      ▼
top 3 trechos
      │
      ▼
[G] assistente.py: Claude responde SÓ com os trechos e cita o id da fonte
      │
      ▼
{ resposta, fontes: [{id, titulo, nota}] }
```

## Decisões técnicas

- **BM25 implementado à mão** (`app/busca.py`), sem biblioteca, para mostrar como a recuperação funciona: frequência do termo, raridade do termo (IDF) e normalização pelo tamanho do trecho.
- **Trechos por seção:** cada `## título` dos arquivos `.md` vira um trecho com id próprio, ex.: `equipamentos#retrolavagem-filtro`.
- **Normalização:** ignora acentos, maiúsculas, palavras vazias ("de", "o", "como") e plural simples.
- **Nota mínima:** se nada for relevante, o assistente não chama a IA e não inventa.
- **Citação obrigatória:** cada resposta mostra de qual documento veio, para o atendente conferir.
- **Teste de qualidade da busca:** perguntas reais com o trecho que deveria ser encontrado. Uma delas falhou na primeira versão ("Quantas horas devo deixar o filtro ligado?") e a correção foi escrever o documento do jeito que o cliente pergunta.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sua_chave   # opcional; sem ela devolve o melhor trecho (modo demo)
uvicorn app.main:app --reload
```

Documentação interativa: http://127.0.0.1:8000/docs

```bash
curl -X POST http://127.0.0.1:8000/perguntas \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Minha água está verde, o que faço?"}'
```

Para ensinar coisas novas ao assistente, basta adicionar um arquivo `.md` em `base_conhecimento/` com seções `## Título`.

## Testes

```bash
pytest -q
```

## Próximos passos

- Trocar BM25 por embeddings + banco vetorial (ex.: pgvector) e comparar a qualidade.
- Busca híbrida (BM25 + embeddings).
- Registrar perguntas sem resposta para descobrir o que falta na base.

> Projeto de portfólio. A loja "Água Azul" e as regras de atendimento são fictícias; as orientações de tratamento da água são gerais e não substituem um técnico.
