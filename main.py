from agent_examples import hello_world
from agent_examples import bank_support

import asyncio


def main():
    # hello_world.run()
    asyncio.run(bank_support.main())



if __name__ == "__main__":
    main()
