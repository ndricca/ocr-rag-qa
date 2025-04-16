import logging
import os

import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from llm_handlers.azure_openai_handler import AzureOpenaiHandler
from llm_handlers.jina_handler import JinaHandler
from utils.conversation_handler import ConversationHandler
from utils.dto import InputMessage
from utils.logger import filter_loggers, LOG_CONFIG
from utils.tool_client import ToolClient

load_dotenv(find_dotenv())

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

logger.info('Starting up...')

# LLM Handlers
llm_handler = AzureOpenaiHandler(chat_model="crif-genai-gpt-4.1-2025-04-14")
jina_handler = JinaHandler(embed_model="jina-clip-v2")

# fake key-value db
conversation_db = {}

tool_client = ToolClient(llm_handler=llm_handler, embeddings_handler=jina_handler,
                         collection_name="9d277797-a704-406c-bd99-a9803d0cf8f5")

logger.info('Starting up... done!')

app = FastAPI()

HTML = """
<!DOCTYPE HTML>
<HTML>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</HTML>
"""


@app.get("/")
async def get():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        logger.info("Waiting for message...")
        data = await websocket.receive_text()
        input_dto = InputMessage(
            conversation_id='a',
            user_id="user_123",
            message=data
        )
        logger.info("Received")
        conversation_handler = ConversationHandler(input_dto,
                                                   conversation_db=conversation_db,
                                                   tool_client=tool_client,
                                                   llm_handler=llm_handler,
                                                   websocket=websocket)
        await conversation_handler.main()

