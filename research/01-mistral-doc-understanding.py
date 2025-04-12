import logging
import os
from argparse import ArgumentParser

from mistralai import Mistral

from utils.logger import filter_loggers

filter_loggers({'httpcore': 'ERROR'})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

parser = ArgumentParser(description="Upload a file to Mistral for OCR processing.")
parser.add_argument('--file_id', type=str, help="File ID to use for OCR")  # 9d277797-a704-406c-bd99-a9803d0cf8f5
parser.add_argument('--model', type=str, help="Model to use for chat completion", default="mistral-small-latest")

QUESTIONS = [
    "In quale stagione è stata inaugurata la Coppa del Mondo di sci alpino?",
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

if __name__ == "__main__":
    args = parser.parse_args()
    file_id = args.file_id
    model = args.model

    try:
        uploaded_pdf = client.files.retrieve(file_id=file_id)
        logger.debug("File retrieved successfully")
    except Exception as e:
        logger.error(f"Error retrieving file with id {file_id}: {e}")
        exit(1)

    # Get the file URL
    signed_url = client.files.get_signed_url(file_id=file_id)

    # Define the messages for the chat
    for question in QUESTIONS:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Il tuo compito è quello di rispondere in maniera chiara e concisa alla domanda, utilizzando il documento fornito come riferimento."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "document_url",
                        "document_url": signed_url.url
                    }
                ]
            }
        ]

        # Get the chat response
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        # Print the content of the response
        answer = chat_response.choices[0].message.content
        logger.info(f'# Question: "{question}"\n# Answer: "{answer}"\n')
