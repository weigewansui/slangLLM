from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import pipeline
from langchain.llms.base import LLM
from huggingface_hub import login
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import torch

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class HuggingFaceHugs(LLM):
  pipeline: Any
  class Config:
    """Configuration for this pydantic object."""
    extra = Extra.forbid

  def __init__(self, model, tokenizer, task="text-generation"):
    super().__init__()
    self.pipeline = pipeline(task, model=model, tokenizer=tokenizer)

  @property
  def _llm_type(self) -> str:
    """Return type of llm."""
    return "huggingface_hub"

  def _call(self, prompt, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,):
    # Runt the inference.
    text = self.pipeline(prompt, max_length=100)[0]['generated_text']

    # @alvas: I've totally no idea what this in langchain does, so I copied it verbatim.
    if stop is not None:
      # This is a bit hacky, but I can't figure out a better way to enforce
      # stop tokens when making calls to huggingface_hub.
      text = enforce_stop_tokens(text, stop)
    print(text)
    return text[len(prompt):]


custom_prompt_template='''Use the following pieces of information to answer the users question. 
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
 prompt =PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

 return prompt

def load_llm():
 n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
 n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
 llm=LlamaCpp(
    model_path='llama-2-7b-chat.Q8_0.gguf',
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=True, # Verbose is required to pass to the callback manager
 )

 return llm


def retrieval_qa_chain(llm,prompt,db):
 qa_chain=RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=db.as_retriever(search_kwargs={'k':2}),
   return_source_documents=True,
   chain_type_kwargs={'prompt':prompt  }
  )
 return qa_chain

def qa_bot():
 embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
  model_kwargs={'device':'cuda'})
 db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
 )
 llm=load_llm()
 qa_prompt=set_custom_prompt()
 qa = retrieval_qa_chain(llm,qa_prompt,db)
 return qa 


def final_result(query):
 qa_result=qa_bot()
 response=qa_result({'query':query})
 return response 

## chainlit here
@cl.on_chat_start
async def start():
 chain=qa_bot()
 msg=cl.Message(content="Firing up SlangLLM...")
 await msg.send()
 msg.content= "Hi, welcome to SlangLLM. What is your query?"
 await msg.update()
 cl.user_session.set("chain",chain)


@cl.on_message
async def main(message):
 chain=cl.user_session.get("chain")
 cb = cl.AsyncLangchainCallbackHandler(
  stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
  )
 cb.ansert_reached=True
 res=await chain.acall(message, callbacks=[cb])
 answer=res["result"]
 sources=res["source_documents"]

 if sources:
  answer+=f"\nSources: "+str(str(sources))
 else:
  answer+=f"\nNo Sources found"

 await cl.Message(content=answer).send() 
