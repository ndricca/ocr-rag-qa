import logging

from mistralai import ToolCall

from llm_handlers.mistral_handler import MistralHandler
from utils.dto import InputMessage, OutputMessage, FIXED_FIELDS
from utils.tool_client import ToolClient

logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    This class handles the conversation with the LLM using an agentic RAG approach.
    """

    def __init__(self,
                 input_dto: InputMessage,
                 conversation_db: dict,
                 llm_handler: MistralHandler,
                 tool_client: ToolClient):
        self.input_dto = input_dto
        self.conversation_db = conversation_db
        self.llm_handler = llm_handler
        self.tool_client = tool_client

        # Initialize the conversation
        self.current_conversation: list[dict] = self.conversation_db.get(self.input_dto.conversation_id, [])

    def update_conversation(self, conversation_id, update_dict: dict | None = None):
        """ Incremental update of the conversation in the database. """
        to_update = {'last_update_timestamp': self.input_dto.timestmap}
        if update_dict:
            to_update = {**to_update, **update_dict}
        # TODO: Uncomment the following line when conversation_db is implemented
        # self.conversation_db.update(conversation_id, to_update)

    def routing_agent(self):
        """
        This method contains the LLM call to the routing agent.
        If the agent return tool calls responses, it handles them and invokes the routing agent again.
        Otherwise, the function ends (assistant responses are stored in current_conversation attribute).
        """
        response = self.llm_handler.invoke_with_retry(
            method="complete",
            messages=self.current_conversation,
            temperature=0.7,
            tools=self.tool_client.tools,
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
        result = self.tool_client.execute(tool_name, tool_call.function.arguments)

        self.current_conversation.append({
            "role": "tool",
            "name": tool_name,
            "content": result,
            "tool_call_id": tool_call_id
        })

    def main(self):
        """
        Main function to handle the conversation. Adds system and user messages to the conversation,
        then invokes the routing agent to process the conversation.
        Finally, it updates the conversation database and returns the output message.
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
        self.routing_agent()
        self.conversation_db[self.input_dto.conversation_id] = self.current_conversation
        output_message = OutputMessage(
            **self.input_dto.model_dump(include=FIXED_FIELDS),
            message=self.current_conversation[-1]["content"],
        )
        return output_message
