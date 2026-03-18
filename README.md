# NLP Data Warehouse Assistant
This project provides a Natural Language to SQL (NL2SQL) interface, allowing users—specifically finance and operations teams—to query a DuckDB data warehouse using plain English. By leveraging Large Language Models (LLMs) and LangChain, the assistant translates questions into optimized SQL, executes them against the trips.duckdb warehouse, and summarizes the findings in natural language.

![Alt Text](https://raw.githubusercontent.com/opadotun-taiwo/NLP_datawarehouse/main/Streamlit%20deployed2.png)

## Quick Start (Docker)
The project is fully containerized. You can launch the Streamlit interface immediately using the pre-built image.

### Bash
docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN=your_token_here nlpwarehousedata
URL: http://localhost:8501

Environment Variables: You can also pass OPENAI_API_KEY if you prefer using GPT models.

### Project Architecture
The system follows a modern data engineering pattern, moving from raw ingestion to an AI-driven consumption layer:

Ingestion: Data is processed and loaded into a local DuckDB instance.

Warehouse: A structured file named trips.duckdb serves as the high-performance analytical engine.

Agent Logic: A LangChain-powered agent handles the "Text-to-SQL" conversion and the "Data-to-Natural-Language" response.

UI: A Streamlit dashboard provides a user-friendly interface for non-technical stakeholders.

### Key Components
agent.ipynb: The original research and development notebook used to prototype the SQL chains and LLM prompts.

app.py: The production-ready Streamlit application.

ingest.py: Script to load parquet or CSV trip data into the DuckDB warehouse.

trips.duckdb: The local data warehouse containing the taxi trip records.

Dockerfile: Defines the environment, dependencies, and entry point for the application.

### Local Development Setup
If you wish to run the project without Docker:

1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Install Dependencies
Bash
pip install streamlit duckdb langchain langchain-openai langchain-huggingface python-dotenv pandas
3. Environment Configuration
Create a .env file in the root directory:

Code snippet
OPENAI_API_KEY=your_openai_key
# OR
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
4. Data Ingestion
If the trips.duckdb file is not present or needs updating, run the ingestion script:

Bash
python ingest.py
5. Launch the App
Bash
streamlit run app.py
Features
Natural Language Querying: "Which day in January had the highest number of taxi trips?"

Schema Awareness: The AI automatically identifies table columns and types to write accurate queries.

Dual-Model Support: Compatible with OpenAI (GPT-4o) and Hugging Face (Qwen/Llama) via HuggingFaceEndpoint.

Transparent Results: Users can expand a "View Raw Data" section to see the generated SQL and the raw dataframe for verification.

### Example Queries
"What was the average trip distance in February?"

"Show me the top 5 pickup locations by total fare amount."

"How many trips were paid for using credit cards vs cash?"
