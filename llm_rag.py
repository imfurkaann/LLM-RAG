import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.chains import RetrievalQA

# OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

pdf_search = None
chat_history = []

def process_file(file):

    loader = read_file(file)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    pdf_search = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    return pdf_search

def read_file(file_path):

    if file_path.endswith(".csv"):
        loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    elif file_path.endswith(".txt") or file_path.endswith(".pdf"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Desteklenmeyen dosya türü!")

    return loader

def generate_response(question, file):

    global pdf_search
    global chat_history
    
    if file:
        pdf_search = process_file(file)

    if not pdf_search:
        return "Lütfen önce bir dosya yükleyin!"

    response = qa().run(question)
    
    chat_history.append({"question": question, "answer": response})
    
    chat_display = "\n".join(
        [f"Soru: {entry['question']}\nCevap: {entry['answer']}" for entry in chat_history]
    )
    
    return chat_display

def qa(persist_dir="./chroma_db"):

    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    llm = OpenAI(temperature=0)

    prompt_template = """
    You are an analysis bot that reviews the information provided in the document and offers insights based on this information. Your role is to help users understand the content, make informed decisions, and plan accordingly. Use only the information provided in the document to answer the question at the end. Filter out results where you are not very sure of the context.
    If the user asks for suggestions, first answer the user directly and then list your suggestions according to the format below. End your answer immediately after giving the suggestions. Placeholders are indicated using [] and comments are indicated using ().

    Answer: 
    [produce short and direct answers by utilizing the information in the document]


    Here is the context:
    {context}

    Question: {question}
    """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    return qa

with gr.Blocks() as interface:
    with gr.Row():
        chat_display = gr.Textbox(label="Chat Ekranı", lines=20, interactive=False)
    
    with gr.Row():
        file = gr.File(label="Dosya Yükle (CSV, TXT, PDF)", type="filepath", interactive=True)
        question = gr.Textbox(label="Sorunuzu girin", lines=2, placeholder="Buraya soruyu yazın...")
        submit_btn = gr.Button("Gönder")
    
    submit_btn.click(fn=generate_response, inputs=[question, file], outputs=chat_display)

interface.launch()
