from typing import List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema.memory import BaseMemory
from langchain.tools import Tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI

from langflow import CustomComponent
from langflow.field_typing.range_spec import RangeSpec


class OpenAIToolAgentComponent(CustomComponent):
    display_name: str = "OpenAI Tool Agent"
    description: str = "Conversational Agent that can use OpenAI's tool calling API"

    def build_config(self):
        openai_tools_models = [
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
        ]

        return {
            "tools": {"display_name": "Tools"},
            "memory": {"display_name": "Memory"},
            "system_message": {"display_name": "System Message"},
            "max_token_limit": {"display_name": "Max Token Limit"},
            "model_name": {
                "display_name": "Model Name",
                "options": openai_tools_models,
                "value": openai_tools_models[0],
            },
            "code": {"show": False},
            "temperature": {
                "display_name": "Temperature",
                "value": 0.2,
                "range_spec": RangeSpec(min=0, max=2, step=0.1),
            },
        }

    def build(
        self,
        model_name: str,
        openai_api_key: str,
        tools: List[Tool],
        openai_api_base: Optional[str] = None,
        memory: Optional[BaseMemory] = None,
        system_message: Optional[SystemMessagePromptTemplate] = None,
        max_token_limit: int = 2000,
        temperature: float = 0.2,
    ) -> AgentExecutor:
        llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            base_url=openai_api_base,
            max_tokens=max_token_limit,
            temperature=temperature,
        )
        _system_message = system_message or SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=[], template="You are a helpful assistant")
        )

        if not memory:
            memory_key = "chat_history"
            memory = ConversationTokenBufferMemory(
                memory_key=memory_key,
                return_messages=True,
                output_key="output",
                llm=llm,
                max_token_limit=max_token_limit,
            )
        else:
            memory_key = memory.memory_key  # type: ignore

        prompt = ChatPromptTemplate.from_messages(
            [
                _system_message,
                MessagesPlaceholder(variable_name=memory_key, optional=True),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(input_variables=["input"], template="{input}")
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,  # type: ignore
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
