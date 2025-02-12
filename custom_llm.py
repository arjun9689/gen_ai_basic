from typing import Any, Dict, Iterator, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration
import requests

class Llama33CustomChat(BaseChatModel):
    api_url: str = "YOUR_COMPANY_API_URL"
    temperature: float = 0.7
    max_tokens: int = 128
    top_p: float = 0.95
    headers: Dict = {
        "Content-Type": "application/json",
        "X-Cluster": "H100",
        "Accept": "*/*"
    }

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to API-compatible format"""
        formatted = []
        for msg in messages:
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                role = "system"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _generate(self, messages: List[BaseMessage], 
                 stop: Optional[List[str]] = None,
                 **kwargs: Any) -> ChatResult:
        
        payload = {
            "model": "/models/Meta-Llama-3.3-8B-Instruct",
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"API Error: {response.text}")

        content = response.json()['choices'][0]['message']['content']
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "llama-3.3-70B-instruct"

    def _stream(self, messages: List[BaseMessage],
               stop: Optional[List[str]] = None,
               **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Implement streaming if API supports it"""
        # Add streaming logic here if supported by your API
        raise NotImplementedError("Streaming not implemented")

-------

from langchain_core.messages import HumanMessage

# Initialize with your API endpoint
model = Llama33CustomChat(api_url="URL")

# Single interaction
response = model.invoke([
    HumanMessage(content="Explain quantum computing")
])
print(response.content)

# Batch processing
batch_responses = model.batch([
    [HumanMessage(content="What is AI?")],
    [HumanMessage(content="Explain blockchain")]
])


-----

from langchain_core.messages import HumanMessage

# Initialize with your API endpoint
model = Llama33CustomChat(api_url="URL")

# Single interaction
response = model.invoke([
    HumanMessage(content="Explain quantum computing")
])
print(response.content)

# Batch processing
batch_responses = model.batch([
    [HumanMessage(content="What is AI?")],
    [HumanMessage(content="Explain blockchain")]
])

----

def _stream(self, messages, **kwargs):
    payload = {**self._create_payload(messages), "stream": True}
    
    with requests.post(self.api_url, headers=self.headers, 
                      json=payload, stream=True) as response:
        for chunk in parse_stream(response):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk['delta'])
            )

------




