TOOLS_OPENAI_SCHEMA = [
            {
                "type": "function",
                "function": {
                    "name": "math_reasoning",
                    "description": "Use this tool to approach the problem using mathematical reasoning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to be answered."
                            },
                            "context": {
                                "type": "string",
                                "description": "Contextual information to help answer the question."
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
            "description": """ Use this tool to search for information in the vector store. Query search expands user input with an hypothetical answer to increase cosine similarity.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Expanded sentence generated from user input to increase cosine similarity."
                    },
                    "limit": {
                        "type": "integer",
                        "description": """Max number of results to return, usually 3-5 are enough. Keep it higher for abstractive queries, lower for extractive (factual) queries.""",
                    }
                },
                "required": ["question"]
            }
        }
    }
]


MATH_REASONING_SYSTEM_TEMPLATE = """Il tuo compito è quello di scrivere il procedimento logico necessario ad ottenere un risultato numerico.
Non concentrarti sul'output finale, ma sul procedimento.

Considera quanto segue per rispondere alla domanda:
{context}
"""