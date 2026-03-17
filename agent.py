#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline


# In[2]:


import duckdb
conn = duckdb.connect("trips.duckdb")


# In[3]:


def get_schema(*args, **kwargs):
    schema = conn.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'trips'
    """).fetchall()

    columns = "\n".join([f"{col} ({dtype})" for col, dtype in schema])

    return f""" Table: trips Columns:{columns}"""

def run_query(query):
    print(f"\nRunning Query:\n{query}\n")
    return conn.execute(query).fetchdf()


# In[4]:


print(get_schema())


# In[5]:


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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to a SQL query. No pre-amble. "
            "Please do not return anything else apart from the SQL query, no prefix aur suffix quotes, no sql keyword, nothing please"),
            ("human", template),
        ]
    )

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


# In[6]:


def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt_response = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
            ),
            ("human", template),
        ]
    )

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


# In[7]:


# Query types to try out:

# 1. Straight forward queries: Give me the name of 10 Artists
# 2. Querying for multiple columns: Give me the name and artist ID of 10 Artists
# 3. Querying a table by a foreign key: Give me 10 Albums by the Artist with ID 1
# 4. Joining with a different table: Give some Albums by the Artist name Audioslave'
# 5. Multi-level Joins: Give some Tracks by the Artist name Audioslave'


# In[8]:


load_dotenv()

# write_sql_query(llm=get_llm(load_from_hugging_face=True)).invoke({"question": "Give me 10 Artists"})
query = 'Which day in January had the highest number of taxi trips?'
response = answer_user_query(query, llm=get_llm(load_from_hugging_face=True))
print(response.content)


# In[9]:


# query = '''
# SELECT T.Name
# FROM Track T
# WHERE T.AlbumId IN (
#     SELECT A.AlbumId
#     FROM Album A
#     LEFT JOIN Artist Ar ON A.ArtistId = Ar.ArtistId
#     WHERE Ar.Name = "Audioslave"
# );
# '''
# query = 'SELECT Album.* FROM Album LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = "Audioslave" LIMIT 10'
query = '''
SELECT EXTRACT(DAY FROM tpep_pickup_datetime) AS trip_day 
FROM trips 
WHERE EXTRACT(MONTH FROM tpep_pickup_datetime) = 1 
GROUP BY tpep_pickup_datetime 
ORDER BY COUNT(*) DESC 
LIMIT 1;
'''
response = run_query(query)
print(response)


# In[ ]:




