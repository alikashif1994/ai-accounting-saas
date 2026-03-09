# src/agents/accounting_agent.py
# LangChain AI Agent — makes autonomous accounting decisions


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from dataclasses import dataclass

load_dotenv()  # Reads your .env file and loads OPENAI_API_KEY

# ── Define the Agent's tools ──────────────────────────────────────────
# Each @tool is something the agent can choose to do.
# The docstring is what the agent reads to understand what the tool does.


@tool
def categorise_and_record(entry_details: str) -> str:
    '''
    Use this tool to officially categorise a financial entry and record it.
    Input: a description of the expense including vendor, amount, and suggested category.
    Use this when the fuzzy confidence score is above 60% and the entry looks legitimate.
    '''
    return f'Entry categorised and recorded: {entry_details}'


@tool
def flag_for_human_review(reason: str) -> str:
    '''
    Use this tool when you are uncertain about an entry or it seems unusual.
    Input: the reason why this entry needs human review.
    Use this when: confidence is below 60%, the amount is unusually large,
    the category is ambiguous, or the vendor is unknown.
    '''
    return f'Entry flagged for human review. Reason: {reason}'


@tool
def check_vat_eligibility(vendor_and_amount: str) -> str:
    '''
    Use this tool to check whether a purchase is eligible for VAT reclaim.
    Input: the vendor name and purchase amount.
    Use this for any business expense where VAT may be reclaimable.
    '''
    return f'VAT eligibility check completed for: {vendor_and_amount}. Standard 20% VAT may be reclaimable if a valid VAT receipt is provided.'


@tool
def request_additional_documents(what_is_needed: str) -> str:
    '''
    Use this tool when the uploaded document is missing key information.
    Input: description of what document or information is needed.
    Use this when: amount is missing, vendor is unidentifiable, date is absent.
    '''
    return f'Request sent to client: please provide {what_is_needed}'




# ── Build the Agent ───────────────────────────────────────────────────
def create_accounting_agent():
    '''Creates and returns the LangChain AI Agent.'''


    # The AI model the agent uses to reason
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,  # 0 = deterministic — same input gives same output (important for audit)
        api_key=os.getenv('OPENAI_API_KEY')
    )


    tools = [
        categorise_and_record,
        flag_for_human_review,
        check_vat_eligibility,
        request_additional_documents
    ]


    # ReAct prompt = Reason + Act. The agent thinks step by step before acting.
    prompt = hub.pull('hwchase17/react')


    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)




@dataclass
class AgentDecision:
    action_taken: str    # Which tool the agent called
    reasoning: str       # Why the agent made that decision
    output: str          # What the tool returned




def make_decision(vendor: str, amount: float, category: str,
                  confidence: float, is_ambiguous: bool) -> AgentDecision:
    '''
    Main function: given all the extracted information, let the agent decide what to do.
    '''
    agent = create_accounting_agent()


    # Construct the context for the agent
    task = f'''
    You are an AI accounting assistant. A client has uploaded a financial document.
    Here are the details extracted from it:


    Vendor: {vendor or 'Unknown'}
    Amount: £{amount:.2f if amount else 'Unknown'}
    Suggested Category: {category or 'Unclassified'}
    Category Confidence: {confidence:.0%}
    Is Ambiguous: {is_ambiguous}


    Decide what action to take using one of your available tools.
    Reason step by step before acting.
    '''


    result = agent.invoke({'input': task})


    return AgentDecision(
        action_taken='Agent decision recorded',
        reasoning=task,
        output=result.get('output', 'Decision recorded')
    )