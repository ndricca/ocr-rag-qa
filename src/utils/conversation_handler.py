import json
import logging
import os

from mistralai import ToolCall
from qdrant_client import QdrantClient

from llm_handlers.mistral_handler import MistralHandler
from utils.dto import InputMessage, OutputMessage, FIXED_FIELDS

logger = logging.getLogger(__name__)


class ConversationHandler:
    def __init__(self, input_dto: InputMessage, conversation_db: dict, collection_name: str = "ski_league"):
        self.input_dto = input_dto
        self.conversation_db = conversation_db
        self.llm_handler = MistralHandler(chat_model="mistral-small-latest")
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_port = os.getenv('QDRANT_PORT')
        self.vector_db = QdrantClient(url=f"{qdrant_url}:{qdrant_port}")
        self.collection_name = collection_name

        # Initialize the conversation
        self.current_conversation: list[dict] = self.conversation_db.get(self.input_dto.conversation_id, [])

        # Set up tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "math_reasoning",
                    "description": "Usa questo strumento per fare ragionamenti logico-matematici.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "La domanda da risolvere."
                            },
                            "context": {
                                "type": "string",
                                "description": "Il contesto da utilizzare per risolvere la domanda."
                            }
                        },
                        "required": ["question", "context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_context",
                    "description": """Usa questo strumento per cercare le informazioni necessarie a rispondere.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "La domanda da risolvere."
                            },
                            "limit": {
                                "type": "integer",
                                "description": """Il numero massimo di risultati da restituire."""
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
        self.tool_to_function = {
            "math_reasoning": self.math_reasoning,
            "get_context": self.get_context
        }

    # def update_conversation(self, conversation_id, update_dict: dict | None = None):
    #     to_update = {'last_update_timestamp': self.input_dto.timestmap}
    #     if update_dict:
    #         to_update = {**to_update, **update_dict}
    #     self.conversation_db.update(conversation_id, to_update)

    def math_reasoning(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"""Il tuo compito è quello di scrivere il procedimento logico necessario ad ottenere un risultato numerico.
Non concentrarti sul'output finale, ma sul procedimento.

Considera quanto segue per rispondere alla domanda:
{context}
"""
            },
            {"role": "user", "content": question},
        ]
        response = self.llm_handler.invoke_with_retry(
            method="complete",
            messages=messages,
        )
        return response.choices[0].message.content

    def get_context(self, question: str, limit: int = 3) -> str:

        results = self.vector_db.query_points(
            collection_name=self.collection_name,
            query_text=question,
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        result_text = sorted(results.points, key=lambda x: x.score, reverse=True)
        return json.dumps(result_text)

    def routing_agent(self):
        """
        This method contains the LLM call to the routing agent.
        If the agent return tool calls responses, it handles them and invokes the routing agent again.
        Otherwise, the function ends (assistant responses are stored in current_conversation attribute).
        """
        if len(self.current_conversation) == 0:
            self.current_conversation.append({
                "role": "system",
                "content": """Il tuo compito è quello di rispondere in maniera chiara e concisa alla domanda, utilizzando i tool a tua disposizione.
Le domande riguarderanno il mondo dello sci, in particolare la Coppa del Mondo di Sci Alpino.
Hai a disposizione un tool per la ricerca di informazioni, e un tool per fare ragionamenti logico-matematici.
Usa il tool di ricerca per ottenere informazioni utili a rispondere alla domanda, e nel caso sia necessario fare calcoli sfrutta il tool di ragionamento matematico.
Se non sei sicuro sulla risposta, chiedi chiarimenti all'utente.
Se la domanda non è pertinente, rispondi ironicamente proponendo una ricetta tipica della Valtellina.
"""
            })

        self.current_conversation.append({
            "role": "user",
            "content": self.input_dto.message,
        })
        response = self.llm_handler.invoke_with_retry(
            method="complete",
            messages=self.current_conversation,
            temperature=0.7,
            tools=self.tools,
            tool_choice="any"
        )
        response_msg = response.choices[0].message
        self.current_conversation.append({
            "role": "assistant",
            "content": response_msg.content,
        })
        if response_msg.tool_calls is not None:
            self.current_conversation[-1]["tool_calls"] = response_msg.tool_calls
            for tool_call in response_msg.tool_calls:
                self.handle_tool_call(tool_call)
                return self.routing_agent()

    def handle_tool_call(self, tool_call: ToolCall):
        """
        Handles the tool call by executing the corresponding function and updating the conversation.

        Parameters:
            tool_call (ToolCall): The tool call object containing the function name and arguments.
        """
        tool_call_id = tool_call.id
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        if tool_name in self.tool_to_function:
            function = self.tool_to_function[tool_name]
            try:
                logger.debug(f"Calling tool {tool_name} with arguments: {tool_args}")
                result = function(**tool_args)
                logger.debug(f"Tool {tool_name} result: {result}")
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
        else:
            result = f"Tool {tool_name} not found in tool_to_function mapping."

        self.current_conversation.append({
            "role": "tool",
            "name": tool_name,
            "content": result,
            "tool_call_id": tool_call_id
        })

    def main(self):
        self.routing_agent()
        self.conversation_db[self.input_dto.conversation_id] = self.current_conversation
        output_message = OutputMessage(
            **self.input_dto.model_dump(include=FIXED_FIELDS),
            role="assistant",
            content=self.current_conversation[-1]["content"],
        )
        return output_message
