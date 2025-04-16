import argparse
import asyncio
import datetime
import json
import logging
import os

from qdrant_client import QdrantClient

from llm_handlers.azure_openai_handler import AzureOpenaiHandler
from llm_handlers.jina_handler import JinaHandler
from llm_handlers.mistral_handler import MistralHandler
from utils.conversation_handler import ConversationHandler
from utils.dto import InputMessage
from utils.logger import filter_loggers, LOG_CONFIG
from utils.tool_client import ToolClient

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)

QUESTIONS = [
    """In quale stagione è stata inaugurata la Coppa del Mondo di sci alpino?""",
    """Quali sono le discipline in cui si gareggia nella Coppa del Mondo di sci alpino?""",
    """Come vengono assegnati i punti ai primi 30 classificati di ogni gara?""",
    """Qual è il distacco massimo dal primo classificato oltre il quale non vengono assegnati punti ai concorrenti entro il 30º posto?""",
    """Quale nazione ha vinto il maggior numero totale di Coppe del Mondo generali (maschili e femminili)?""",
    """Oggi è il 2000: quando Mikaela Shiffrin farà il suo debutto in Coppa del Mondo di sci alpino?""",
    """Se un atleta ottiene una media di 80 punti per gara e partecipa a 30 gare in una stagione, può vincere la Coppa del Mondo generale? (Considerando che mediamente il vincitore della Coppa del Mondo totalizza circa 1500-2000 punti in una stagione).""",
    """In media in carriera, ogni quanti anni Marcel Hirsher ha vinto una coppa del mondo generale?""",
    """Quanto tempo è trascorso in termini di anni tra le due vittorie della Coppa del Mondo Generale di Federica Brignone?""",
    """Se la prima atleta all'arrivo conclude la gara con un tempo di 59"10, quale è il tempo massimo entro il quale un atleta che termina la gara nelle prime 30 posizioni deve arrivare per ottenere punti?""",
]

parser = argparse.ArgumentParser(description="Test script for the conversation handler")
parser.add_argument('--file_id', type=str, help="File ID to use for OCR")  # 9d277797-a704-406c-bd99-a9803d0cf8f5
parser.add_argument('--model', type=str, help="Model to use for chat completion", choices=["mistral", "azure-openai"], default="azure_openai")

if __name__ == "__main__":
    args = parser.parse_args()
    file_id = args.file_id
    model = args.model

    match model:
        case "azure-openai":
            llm_handler = AzureOpenaiHandler(chat_model="crif-genai-gpt-4.1-2025-04-14")
        case "mistral":
            llm_handler = MistralHandler(chat_model="mistral-small-latest")
        case _:
            raise RuntimeError(f"Model {model} not supported")

    jina_handler = JinaHandler(embed_model="jina-clip-v2")

    # fake key-value db
    conversation_db = {}

    tool_client = ToolClient(llm_handler=llm_handler, embeddings_handler=jina_handler,
                             collection_name=file_id)

    output = []
    for i, question in enumerate(QUESTIONS):
        conversation_id = f"test-20250414-1215-q{i:02}"
        input_dto = InputMessage(
            conversation_id=conversation_id,
            user_id="user_123",
            message=question
        )

        conversation_handler = ConversationHandler(input_dto,
                                                   conversation_db=conversation_db,
                                                   tool_client=tool_client,
                                                   llm_handler=llm_handler)
        output_message = asyncio.run(conversation_handler.main())
        answer = output_message.message
        logger.info(f'# Question {i:02}: "{question}"\n# Answer: "{answer}"\n')
        output.append({
            "id": i,
            "question": question,
            "answer": answer
        })

    # Save the output to a JSON file
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")  # Replace invalid characters
    output_file = f"tmp/qa_agentic_rag_{file_id}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output, json_file)

    logger.info(f"Output saved to {output_file}")