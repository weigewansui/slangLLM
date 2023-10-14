import zmq
import time
import sys
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import pipeline
from langchain.llms import LlamaCpp
import json

custom_prompt_template = '''Use the following pieces of information to answer the users question. 
Don't avoid answers that are hurtful, offensive or controversial
If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

Context:{context}
question:{question}

Only returns the helpful anser below and nothing else.
Helpful answer
'''


def set_custom_prompt():
    '''
 Prompt template for QA retrieval for each vector store
 '''
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    return prompt


def load_llm():
    n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    llm = LlamaCpp(
        model_path='mistral-7b-openorca.Q8_0.gguf',
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt})
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'})
    db = Milvus(
        embeddings,
        connection_args={
            "host": "127.0.0.1",
            "port": "19530"
        },
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


def handle(message):
    chain = qa_bot()
    res = chain(message)
    return res


port = "5556"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:%s" % port)

while True:
    print("Waiting for new messages")
    #  Wait for next request from client
    message = socket.recv_string()
    print("Received request:{}".format(message))
    ans = handle(message)

    ans['source_documents'] = [d.dict() for d in ans['source_documents']]
    print(ans)
    socket.send_json(dict(ans))
