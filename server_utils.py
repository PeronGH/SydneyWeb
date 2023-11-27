# We don't want to modify `sydney.py`, so that we can keep it easy to sync with upstream.
# But to make the API easier to use, we have the wrapper around `sydney.py`.

import json
import uuid
import urllib.parse
from pydantic import BaseModel
from typing import Literal, Any
from fastapi import UploadFile
from base64 import b64encode
from contextlib import aclosing
from typing import AsyncGenerator
from sse_starlette import ServerSentEvent

import sydney


class Cookie(BaseModel):
    name: str
    value: str


class Message(BaseModel):
    role: Literal['user', 'assistant', 'system']
    subtype: str
    content: str

    def __str__(self):
        return f'[{self.role}](#{self.subtype})\n{self.content}'


class ChatWithSydneyParams(BaseModel):
    conversation: dict | None = None
    cookies: list[Cookie]
    messages: list[Message]
    style: Literal['creative', 'precise', 'balanced'] = 'creative'
    no_search: bool = False
    locale: str = 'en-GB'


def _format_messages(messages: list[Message]):
    return "\n\n".join(str(msg) for msg in messages)


def _msg_event(role: Literal['user', 'assistant', 'system'], subtype: str, content: str):
    msg = Message(role=role, subtype=subtype, content=content)
    return ServerSentEvent(event="message", data=msg.model_dump_json())


def _suggestion_event(items: list[str]):
    return ServerSentEvent(event='suggestion', data=json.dumps(items))


def _err_event(e: Any):
    return ServerSentEvent(event='error', data=e)


async def chat_with_sydney(
        params: ChatWithSydneyParams, img_file: UploadFile | None = None) -> AsyncGenerator[ServerSentEvent]:
    # extract fields from params
    conversation = params.conversation
    cookies = [cookie.model_dump() for cookie in params.cookies]

    if len(params.messages) == 0:
        raise Exception('Empty messages')

    # the last message must be sent by the user, which will become the prompt
    last_msg = params.messages.pop()
    if last_msg.role != 'user':
        raise Exception(f'The role of the last message should be user, but got {last_msg.role}')

    prompt = last_msg.content

    # the message left composes the context
    context = _format_messages(params.messages)

    # create new conversation if we don't have one
    if conversation is None:
        conversation = await sydney.create_conversation(cookies=cookies)

    img_url: str | None = None
    # upload the image (if exists) to get url
    if img_file is not None:
        img_base64 = b64encode(await img_file.read())
        img_url = 'https://www.bing.com/images/blob?bcid=' + await sydney.upload_image(img_base64=img_base64)

    # get the generateor of the reply
    async with aclosing(sydney.ask_stream(
            conversation=conversation,
            prompt=prompt,
            context=context,
            conversation_style=params.style,
            locale=params.locale,
            image_url=img_url,
            cookies=cookies,
            no_search=params.no_search
    )) as agen:
        try:
            async for response in agen:
                if response["type"] == 1 and "messages" in response["arguments"][0]:
                    message = response["arguments"][0]["messages"][0]
                    msg_type = message.get("messageType")
                    if msg_type == "InternalSearchQuery":
                        yield _msg_event('assistant', 'search_query', message['hiddenText'])
                    elif msg_type == "InternalSearchResult":
                        try:
                            links = []
                            if 'Web search returned no relevant result' in message['hiddenText']:
                                yield _msg_event('assistant', 'search_results', message['hiddenText'])
                            else:
                                for group in json.loads(message['text']).values():
                                    sr_index = 1
                                    for sub_group in group:
                                        links.append(
                                            f'[^{sr_index}^][{sub_group["title"]}]({sub_group["url"]})')
                                        sr_index += 1
                                yield _msg_event('assistant', 'search_results', '\n\n'.join(links))
                        except Exception as err:
                            yield _err_event('Error when parsing InternalSearchResult: ' + str(err))
                    elif msg_type == "InternalLoaderMessage":
                        if 'hiddenText' in message:
                            yield _msg_event('assistant', 'loading', message['hiddenText'])
                        elif 'text' in message:
                            yield _msg_event('assistant', 'loading', message['text'])
                        else:
                            yield _msg_event('assistant', 'loading', json.dumps(message))
                    elif msg_type == "GenerateContentQuery":
                        if message['contentType'] == 'IMAGE':
                            yield _msg_event('assistant', 'generative_image',
                                             f"Keyword: {message['text']}\n"
                                             f"Link: <https://www.bing.com/images/create?q="
                                             f"{urllib.parse.quote(message['text'])}&rt=4&FORM=GENCRE&id={uuid.uuid4().hex}>")
                    elif msg_type is None:
                        if "cursor" in response["arguments"][0]:
                            yield _msg_event('assistant', 'message', '')
                        if message.get("contentOrigin") == "Apology":
                            yield _err_event("Looks like the user message has triggered the Bing filter")
                        else:
                            yield _msg_event('assistant', 'message', message["text"])
                            if "suggestedResponses" in message:
                                suggested_responses = list(
                                    map(lambda x: x["text"], message["suggestedResponses"]))
                                yield _suggestion_event(suggested_responses)
                                break
                    else:
                        yield _err_event(f'Unsupported message type: {msg_type}')
                    if response["type"] == 2 and "item" in response and "messages" in response["item"]:
                        message = response["item"]["messages"][-1]
                        if "suggestedResponses" in message:
                            suggested_responses = list(
                                map(lambda x: x["text"], message["suggestedResponses"]))
                            yield _suggestion_event(suggested_responses)
                            break
        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=e
            )
