import streamlit as st
import duckdb
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# Setup Streamlit page configuration
st.set_page_config(page_title="Finance Data Assistant", layout="centered")

# Cache the database connection so it's not recreated on every interaction
@st.cache_resource
def get_db_connection():
    # Connecting in read_only mode is generally safer for user-facing query apps
    return duckdb.connect("trips.duckdb", read_only=True)

conn = get_db_connection()

def get_schema(*args, **kwargs):
    schema = conn.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'trips'
    """).fetchall()

    columns = "\n".join([f"{col} ({dtype})" for col, dtype in schema])
    return f""" Table: trips Columns:\n{columns}"""


def clean_sql(query: str) -> str:
    # Remove markdown code blocks like ```sql ... ```
    query = re.sub(r"```sql|```", "", query, flags=re.IGNORECASE).strip()
    return query

def run_query(query):
    cleaned_query = clean_sql(query)
    print(f"\nRunning Query:\n{cleaned_query}\n")
    return conn.execute(cleaned_query).fetchdf()

def get_llm(load_from_hugging_face=False):
    if load_from_hugging_face:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            task="text-generation",
            provider="hyperbolic",  # set your provider here
        )
        return ChatHuggingFace(llm=llm)
    
    return ChatOpenAI(model="gpt-4", temperature=0.0)

def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given an input question, convert it to a SQL query. No pre-amble. "
                   "Please do not return anything else apart from the SQL query, no prefix or suffix quotes, no sql keyword, nothing please"),
        ("human", template),
    ])

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt_response = ChatPromptTemplate.from_messages([
        ("system", "Given an input question and SQL response, convert it to a natural language answer. No pre-amble."),
        ("human", template),
    ])

    full_chain = (
        RunnablePassthrough.assign(query=write_sql_query(llm))
        | RunnablePassthrough.assign(
            schema=lambda _: get_schema(),
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | llm
    )

    return full_chain.invoke({"question": query})

# --- Streamlit UI ---

st.title("Operations Data Warehouse Assistant")
st.markdown("Ask natural language questions to query the datawarehouse to get updates on the app operations.")

# UI Elements
user_query = st.text_area("What would you like to know?", placeholder="e.g., Which day in January had the highest number of taxi trips?")

# You can toggle this to True if you want to default to the open-source model
USE_HUGGING_FACE = True 

if st.button("Run Query"):
    if not user_query.strip():
        st.warning("Please enter a question before running.")
    else:
        with st.spinner("Analyzing data and generating response..."):
            try:
                # Initialize LLM
                llm = get_llm(load_from_hugging_face=USE_HUGGING_FACE)
                
                # Get the natural language response
                response = answer_user_query(user_query, llm)
                
                # Display the result
                st.success("Query Complete!")
                st.markdown("### Answer")
                st.write(response.content)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")