import chainlit as cl
import zmq
import json
import asyncio

context = zmq.Context()


## chainlit here
@cl.on_chat_start
async def start():
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5556")
    cl.user_session.set("socket", socket)
    msg = cl.Message(content="Firing up SlangLLM...")
    await msg.send()
    msg.content = "Hi, welcome to SlangLLM. What is your query?"
    await msg.update()


async def wait_for_msg(socket, message):
    #  Socket to talk to server
    socket.send_string(message)
    loop = asyncio.get_running_loop()
    res_str = await loop.run_in_executor(None, socket.recv_string)
    res = json.loads(res_str)
    return res


@cl.on_message
async def main(message):
    socket = cl.user_session.get("socket")
    res = await wait_for_msg(socket, message)
    answer = res["result"]
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources: " + str(str(sources))
    # else:
    #     answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()


@cl.on_chat_end
def end():
    socket = cl.user_session.get("socket")
    socket.disconnect("tcp://127.0.0.1:5556")
