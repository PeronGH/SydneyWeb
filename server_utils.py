# We don't want to modify `sydney.py`, so that we can keep it easy to sync with upstream.
# But to make the API easier to use, we have the wrapper around `sydney.py`.

from pydantic import BaseModel
from typing import Literal
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
    type: str
    content: str

    def __str__(self):
        return f'[{self.role}](#{self.type})\n{self.content}'


class ChatWithSydneyParams(BaseModel):
    conversation: dict | None = None
    cookies: list[Cookie]
    messages: list[Message]
    style: Literal['creative', 'precise', 'balanced'] = 'creative'
    no_search: bool = False
    locale: str = 'en-GB'


def _format_messages(messages: list[Message]):
    return "\n\n".join(str(msg) for msg in messages)


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
            async for msg in agen:
                yield ServerSentEvent(
                    event="message",
                    data=msg
                )
        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=e
            )
