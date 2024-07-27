from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


supervisor_system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  [Researcher, Coder]. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", supervisor_system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? "
               "Select one of: [Researcher, Coder, Finished] without any other words")
])

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a web researcher."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You may generate safe python code to analyze data and generate charts using matplotlib."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
