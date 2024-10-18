from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(), multiply]
    # llm = ChatOpenAI(model="gpt-4-turbo")
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    res = agent_executor.invoke(
        {
            "input": "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious",
        }
    )

    print(res)
