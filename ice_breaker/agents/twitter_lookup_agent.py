import sys

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

sys.path.append("/home/roland-szogi/Documents/langchain/ice_breaker")

from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    template = """
       given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username
       In Your Final answer only the person's username
       which is extracted from: https://x.com/USERNAME"""
    tools_for_agent_twitter = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Twitter Page URL",
        ),
    ]

    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=template)

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent_twitter, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent_twitter, verbose=True)

    result = agent_executor.invoke(input={"input": prompt_template.format_prompt(name_of_person=name)})

    twitter_username = result["output"]

    return twitter_username


if __name__ == "__main__":
    linkedin_url = lookup(name="Eden Marco")
    print(linkedin_url)
