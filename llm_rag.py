import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.chains import RetrievalQA

class DocumentProcessor:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.pdf_search = None
    
    def process_file(self, file):
        loader = self.read_file(file)
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        self.pdf_search = Chroma.from_documents(documents, embeddings, persist_directory=self.persist_dir)
        return self.pdf_search

    def read_file(self, file_path):
        if file_path.endswith(".csv"):
            loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
        elif file_path.endswith(".txt") or file_path.endswith(".pdf"):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Desteklenmeyen dosya türü!")
        return loader

class QA:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        self.llm = OpenAI(temperature=0)
        self.qa_chain = self.create_qa_chain()
    
    def create_qa_chain(self):
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
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),
            chain_type_kwargs=chain_type_kwargs
        )

        return qa

    def get_response(self, question):
        response = self.qa_chain.run(question)
        return response

class Chatbot:
    def __init__(self):
        self.chat_history = []
        self.document_processor = DocumentProcessor()
        self.qa_system = QA()
    
    def generate_response(self, question, file):
        if file:
            self.document_processor.process_file(file)

        if not self.document_processor.pdf_search:
            return "Lütfen önce bir dosya yükleyin!"

        response = self.qa_system.get_response(question)
        self.chat_history.append({"question": question, "answer": response})
        
        chat_display = "\n".join(
            [f"Soru: {entry['question']}\nCevap: {entry['answer']}" for entry in self.chat_history]
        )
        
        return chat_display

# Gradio interface
def run_gradio_interface():
    chatbot = Chatbot()

    with gr.Blocks() as interface:
        with gr.Row():
            chat_display = gr.Textbox(label="Chat Ekranı", lines=20, interactive=False)

        with gr.Row():
            file = gr.File(label="Dosya Yükle (CSV, TXT, PDF)", type="filepath", interactive=True)
            question = gr.Textbox(label="Sorunuzu girin", lines=2, placeholder="Buraya soruyu yazın...")
            submit_btn = gr.Button("Gönder")

        submit_btn.click(fn=chatbot.generate_response, inputs=[question, file], outputs=chat_display)

    interface.launch()

if __name__ == "__main__":
    # OpenAI API key
    os.environ["OPENAI_API_KEY"] = "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"
    
    run_gradio_interface()
