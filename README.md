# Paper Pilot

Paper Pilot is an AI-powered research assistant that helps you analyze academic papers and conduct literature reviews. Whether you have specific PDFs to analyze or want to explore the latest research on a topic, Paper Pilot makes academic research more accessible and efficient.

## What Paper Pilot Does

- **PDF Analysis**: Upload your research papers and get intelligent summaries, key findings, and insights
- **ArXiv Discovery**: Enter keywords or research questions to automatically find and analyze recent papers from ArXiv
- **Conversational Interface**: Ask follow-up questions and dive deeper into specific aspects of the research

## Key Features

- **Smart Query Processing**: Transforms your research questions into effective academic search terms
- **Automated Paper Retrieval**: Fetches the most recent and relevant papers from ArXiv based on your interests
- **Memory-Enabled Conversations**: Remembers context from previous questions for more natural interactions
- **Domain Agnostic**: Works across all academic fields - from computer science to biology to social sciences

## How It Works

Paper Pilot operates in two modes:

### PDF Mode
Upload your research papers and Paper Pilot will:
- Extract and analyze the content
- Identify key findings and methodologies
- Answer specific questions about the papers

### ArXiv Mode  
Provide keywords or research questions and Paper Pilot will:
- Generate relevant academic search terms
- Fetch recent papers from ArXiv
- Highlight emerging trends and breakthrough research

## Prerequisites

- Python 3.8+
- Cohere API key (get one at [cohere.ai](https://dashboard.cohere.com/api-keys))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hemhemoh/paper_pilot
cd paper_pilot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Cohere API key:
```bash
export COHERE_API_KEY="your-api-key-here"
```

## Usage

The system processes research papers and provides intelligent analysis through a conversational interface. You can either upload PDF files for analysis or use keywords to discover and analyze recent ArXiv publications.

## Example Use Cases

- **Literature Review**: "What are the latest developments in transformer architectures?"
- **Paper Summary**: Upload a complex research paper and ask "What are the main contributions of this work?"
- **Comparative Analysis**: "How do the methodologies in these papers compare?"

## Dependencies

- **langchain**: Core framework for building the RAG pipeline
- **langchain_cohere**: Cohere integration for embeddings and chat
- **langchain_docling**: PDF processing and document loading
- **feedparser**: ArXiv API integration
- **faiss-cpu**: Vector similarity search
- **qdrant-client**: Alternative vector database

Paper Pilot transforms the way you interact with academic literature, making research more efficient and insights more accessible.
