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

import zmq
import json

context = zmq.Context()
socket = None
socket = context.socket(zmq.REQ)


## chainlit here
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Firing up SlangLLM...")
    await msg.send()
    msg.content = "Hi, welcome to SlangLLM. What is your query?"
    await msg.update()


async def wait_for_msg(socket, message):
    socket.send_string(message)
    res_str = socket.recv_string()
    res = json.loads(res_str)
    return res


@cl.on_message
async def main(message):
    #  Socket to talk to server
    socket.connect("tcp://127.0.0.1:5556")

    res = await wait_for_msg(socket, message)
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()
