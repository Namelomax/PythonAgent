import json
import os
from dotenv import load_dotenv

from DocumentAssistant.document_assistant import DocumentAssistant


def main():
    load_dotenv()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    assistant = DocumentAssistant(
        chunk_size=500,
        overlap=50,
        top_k=3,
    )

    documents = [
        "data/A9RD3D4.pdf",
        "data/Polzovatelskoe_soglashenie.pdf",
        "data/University Success.docx",
    ]

    print("Indexing documents...")
    assistant.index_documents(documents)
    print(f"Indexed {len(assistant.chunks)} chunks\n")

    questions = [
        "О чем говорится в пользовательском соглашении?",
        "Какие рекомендации даются студентам для успешного обучения?",
        "Какие ограничения или обязанности описаны в документах?",
    ]

    results = []

    for q in questions:
        print(f"Question: {q}")
        answer = assistant.answer_query(q)
        print(f"Answer: {answer}\n")

        results.append({
            "question": q,
            "answer": answer,
        })

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Results saved to results.json")


if __name__ == "__main__":
    main()
