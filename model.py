from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.cache import SQLiteCache
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
You are an expert in homeopathy. Based on the provided context, answer the user's question concisely and only provide the information requested.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[-max_tokens:])
    return text

# Use this when preparing context before feeding it to the model


def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})  # Fetches only 1 document to minimize context size
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,  # Use the correct retriever object
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=120,
        temperature=0.6,
        top_p=0.9
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def remove_redundant_phrases(text):
    # Basic approach to remove near-duplicate sentences
    sentences = text.split('. ')
    seen_sentences = set()
    filtered_sentences = []
    for sentence in sentences:
        if sentence not in seen_sentences:
            filtered_sentences.append(sentence)
            seen_sentences.add(sentence)
    return '. '.join(filtered_sentences)

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    
    if isinstance(response["result"], list):
        answers = [answer for answer in response["result"] if answer not in [None, ""]]
        answer = "\n".join(answers) if answers else "No answers found"
    else:
        answer = response["result"]
    
    # Remove redundant phrases
    answer = remove_redundant_phrases(answer)
    return answer


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    answer = " ".join(dict.fromkeys(answer.split()))
    await cl.Message(content=answer).send()
