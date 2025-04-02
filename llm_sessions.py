import time
from copy import deepcopy
from http import HTTPStatus
import openai

from ga_src.env import APIUS_SLEEP, GPT, GEMINI, QWEN, GLM, CLAUDE, GPT4o


def common_session(client, messages, model_name, temperature, max_token, add=None, _msg=None, tm_slp=30):
    try:
        time.sleep(tm_slp)
        new_messages = deepcopy(messages)
        if _msg is not None:
            new_messages.append({"role": "user", "content": _msg})
        # print(f'session: {_msg}')
        res = client.chat.completions.create(
            model=model_name,
            messages=new_messages,
            temperature=temperature,
            max_tokens=max_token,
        )
        infomation = res.choices[0].message
        new_messages.append(infomation)
        if add is not None:
            messages = new_messages
        return infomation, messages

    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
        time.sleep(APIUS_SLEEP)
        return common_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
        time.sleep(APIUS_SLEEP)
        return common_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
        time.sleep(APIUS_SLEEP)
        return common_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
    except BaseException as e:
        print(f'\n{e}\n')
        print(f'unknown api error occurs, sleep {APIUS_SLEEP} seconds.\n')
        time.sleep(APIUS_SLEEP)
        return common_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)

# def qwen_session(client, messages, model_name, temperature, max_token, add=None, _msg=None, tm_slp=30):
#     try:
#         time.sleep(tm_slp)
#         new_messages = deepcopy(messages)
#         if _msg is not None:
#             new_messages.append({"role": "user", "content": _msg})
#         # print(f'session: {_msg}')
#
#         res = client.chat.completions.create(
#             model=model_name,
#             messages=new_messages,
#             temperature=temperature,
#             max_tokens=max_token,
#         )
#         infomation = res.choices[0].message
#         new_messages.append(infomation)
#         if add is not None:
#             messages = new_messages
#         return infomation, messages
#
#     except openai.APIConnectionError as e:
#         # Handle connection error here
#         print(f"Failed to connect to OpenAI API: {e}")
#         print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
#         time.sleep(APIUS_SLEEP)
#         return qwen_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
#     except openai.RateLimitError as e:
#         # Handle rate limit error (we recommend using exponential backoff)
#         print(f"OpenAI API request exceeded rate limit: {e}")
#         print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
#         time.sleep(APIUS_SLEEP)
#         return qwen_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
#     except openai.APIError as e:
#         # Handle API error here, e.g. retry or log
#         print(f"OpenAI API returned an API Error: {e}")
#         print(f'api error occurs, sleep {APIUS_SLEEP} seconds.\n')
#         time.sleep(APIUS_SLEEP)
#         return qwen_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)
#     except BaseException as e:
#         print(f'\n{e}\n')
#         print(f'unknown api error occurs, sleep {APIUS_SLEEP} seconds.\n')
#         time.sleep(APIUS_SLEEP)
#         return qwen_session(client, messages, model_name, temperature, max_token, add=add, _msg=_msg, tm_slp=tm_slp)

my_sessions = {
    GPT: common_session,
    GEMINI: common_session,
    QWEN: common_session,
    GLM: common_session,
    GPT4o: common_session,
    CLAUDE: common_session,
}