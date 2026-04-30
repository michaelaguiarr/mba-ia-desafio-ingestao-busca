import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
POSTGRES_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

_faltando = [nome for nome, valor in {
    "PDF_PATH": PDF_PATH,
    "POSTGRES_URL": POSTGRES_URL,
    "PG_VECTOR_COLLECTION_NAME": COLLECTION_NAME,
}.items() if not valor]
if _faltando:
    print(f"Erro: variáveis de ambiente não definidas: {', '.join(_faltando)}")
    print("Verifique se o arquivo .env existe e contém as variáveis necessárias.")
    exit(1)


def ingest_pdf():
    # Carrega o PDF página por página
    loader = PyPDFLoader(PDF_PATH)
    documentos = loader.load()

    # Divide em chunks de 1000 chars com overlap de 150
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documentos)

    # Gera embeddings e persiste no PostgreSQL+pgvector
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=POSTGRES_URL,
    )

    print(f"{len(chunks)} chunks ingeridos com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
