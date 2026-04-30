# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF, armazena embeddings no PostgreSQL+pgvector e responde perguntas com base no conteúdo do documento via OpenAI.

## Pré-requisitos

- Docker e Docker Compose
- Python 3.10+
- Chave de API da OpenAI

## Como executar

**1.** Clone o repositório e entre na pasta do projeto.

**2.** Copie o arquivo de variáveis de ambiente e preencha a `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

**3.** Crie e ative o ambiente virtual:

```bash
python3 -m venv venv && source venv/bin/activate
```

**4.** Instale as dependências:

```bash
pip install -r requirements.txt
```

**5.** Suba o banco de dados:

```bash
docker compose up -d
```

**6.** Execute a ingestão do PDF (apenas uma vez):

```bash
python src/ingest.py
```

**7.** Inicie o chat:

```bash
python src/chat.py
```

Digite sua pergunta e pressione Enter. Para encerrar, digite `sair` ou pressione `Ctrl+C`.
