from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from env import OPENAI_API_KEY

def run():
    model = OpenAIModel("gpt-4o-mini",
                        provider=OpenAIProvider(api_key=OPENAI_API_KEY))
    agent = Agent(model, system_prompt='Be concise, reply with one sentence.')

    result = agent.run_sync('Where does "hello world" come from?')  


    print(result.data)
