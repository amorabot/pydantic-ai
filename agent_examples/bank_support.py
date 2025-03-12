from dataclasses import dataclass

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import asyncio

from env import OPENAI_API_KEY


# A basic class to represent a client, aiming to simulate a real client object and db queries
class MockClient:
    name: str
    balance: float
    pending: float

    #  @classmethod -> static method

    def __init__(self, client_name: str, balance: float, pending: float):
        self.name = client_name
        self.balance = balance
        self.pending = pending

    async def get_balance(self, include_pending: bool = False) -> float:
        if include_pending:
            return self.balance + self.pending
        return self.balance
    
    async def get_name(self) -> str:
        return self.name

class MockDatabase:
    """
    Mock database for testing purposes.
    """
    open_state: bool = False
    database: dict[int, MockClient] = {}

    def open(self):
        self.open_state = True
    def close(self):
        self.open_state = False

    def add_client(self, client_id: int, client: MockClient):
        if not self.open_state:
            raise Exception("Database is not open")
        self.database[client_id] = client

    def get_client(self, client_id: int) -> MockClient:
        if not self.open_state:
            raise Exception("Database is not open")
        if not client_id in self.database:
            raise Exception(f"Client not found: {client_id}")
        return self.database[client_id]

@dataclass
class SupportAgentDependencies:
    """
    Dependencies for 
    """
    client_id: int
    database: MockDatabase
    
    def __init__(self, client_id: int, database: MockDatabase):
        self.client_id = client_id
        self.database = database



class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their card or not')
    risk: int = Field(description='Risk level of query', ge=0, le=10)

db = MockDatabase()

db.open()
client_john = MockClient(client_name="John Doe", balance=1000, pending=12.4)
db.add_client(1, client_john)
client_mel = MockClient(client_name="Melissa Rayes", balance=69, pending=100)
db.add_client(2, client_mel)
db.close()

# Agent creation
model = OpenAIModel("gpt-4o-mini",
                    provider=OpenAIProvider(api_key=OPENAI_API_KEY))
support_agent = Agent(model,
                      deps_type=SupportAgentDependencies,
                      result_type=SupportResult,  
                      system_prompt=(  
                        'You are a support agent in our bank, give the '
                        'customer support and judge the risk level of their query.'
                    ))

# Agents tools, and general configs
@support_agent.system_prompt  
async def add_customer_name(ctx: RunContext[SupportAgentDependencies]) -> str:
    db_ref = ctx.deps.database
    db_ref.open()
    customer_name = await db_ref.get_client(client_id=ctx.deps.client_id).get_name()
    db_ref.close()
    return f"The customer's name is {customer_name!r}"

@support_agent.tool
async def customer_balance(ctx: RunContext[SupportAgentDependencies], include_pending: bool) -> float:
    """Returns the customer's current account balance."""  
    db_ref = ctx.deps.database
    db_ref.open()
    balance = await db_ref.get_client(client_id=ctx.deps.client_id).get_balance(include_pending=include_pending)
    db_ref.close()
    return balance



async def main():
    dependencies_mel = SupportAgentDependencies(client_id=2, database=db)
    result_a = await support_agent.run('What is my balance? Ignore pending transactions.', deps=dependencies_mel)
    print(result_a.data)
    result_b = await support_agent.run('What is my balance?', deps=dependencies_mel)
    print(result_b.data)

    dependencies_john = SupportAgentDependencies(client_id=1, database=db)
    result_c = await support_agent.run('I just lost my card!', deps=dependencies_john)
    print(result_c.data)


if __name__ == "__main__":
    asyncio.run(main())