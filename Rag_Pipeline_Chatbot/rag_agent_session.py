from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import streamlit as st
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


if 'document_uploaded' not in st.session_state:
  st.session_state.document_uploaded = False
  
if 'agent' not in st.session_state:
  st.session_state.agent = None
  
if 'vector_store' not in st.session_state:
  st.session_state.vector_store = None
  
if 'messages' not in st.session_state:
  st.session_state.messages = []
  

def process_document(path):
  loader = PyPDFDirectoryLoader(path)
  docs = loader.load()

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
  )

  chunks = splitter.split_documents(docs)

  embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

  vector_db = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
  )

  @tool
  def retriever_tool(query:str):
    """This tool is help you to retrieve the relevant data of the PDF documents, and these PDF documents have the details of the medical report.

    Args:
      query (str): User question
        
    Returns:
      str : Most relevant or similar solution or answer.
    """
    
    data = vector_db.similarity_search(query=query, k=3)
    
    context = ""
    
    for doc in data:
      context += doc.page_content + "\n\n"
      
    return context

  llm = ChatGroq(model='openai/gpt-oss-20b')

  system_prompt = """
    You are a helpful assistant that answers questions ussing retrieved context.
    ALWAYS use a 'retriever_tool' tool for questions requiring external.
    Do not provide more details just give point to pint answers for questions.
  """

  memory = InMemorySaver()

  agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt=system_prompt,
    checkpointer=memory
  )
  
  st.session_state.agent = agent
  st.session_state.document_uploaded = True


# file uploaded/ upload ui
if not st.session_state.document_uploaded:
  uploaded = st.file_uploader(label="Select PDF Files", type=['pdf'], accept_multiple_files=True)
  if uploaded:
    with st.spinner("Processing...."):
      path = './doc_files/'
      for file in uploaded:
        with open(path + file.name, 'wb') as f:
          f.write(file.getvalue())

      process_document(path)
      st.rerun()
      
      
# Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
  query = st.chat_input("Ask Anything?")
  
  for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)      

  if query:
    st.chat_message('user').markdown(query)
    st.session_state.messages.append({'role':'user', 'content' : query})
    
    with st.chat_message('ai'):
      response = st.write_stream(chunk[0].content for chunk in st.session_state.agent.stream(
        {'messages' : [{'role' : 'user', 'content' : query}]},
        {'configurable' : {'thread_id' : '1'}},
        stream_mode='messages'
      ))
      
    st.session_state.messages.append({'role':'ai', 'content' : response})