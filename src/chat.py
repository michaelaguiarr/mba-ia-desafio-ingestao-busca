from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from search import search_prompt, PROMPT_TEMPLATE

load_dotenv()


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    print("Chat iniciado. Digite 'sair' ou pressione Ctrl+C para encerrar.\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if pergunta.lower() == "sair":
            print("Encerrando...")
            break

        if not pergunta:
            continue

        # Busca os chunks mais relevantes no banco vetorial
        resultados = search_prompt(pergunta)
        contexto = "\n\n".join(doc.page_content for doc, _ in resultados)

        # Monta o prompt com o contexto recuperado e chama o LLM
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
        resposta = llm.invoke([HumanMessage(content=prompt)])
        print(f"\nRESPOSTA: {resposta.content}\n")


if __name__ == "__main__":
    main()
