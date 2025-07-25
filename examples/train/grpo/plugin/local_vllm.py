import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI

from dotenv import load_dotenv
load_dotenv()

h20_93_url='https://u381308-9341-dafb4f7f.bjc1.seetacloud.com:8443/v1'


class LocalLLM:
    _qwen3_14b_awq_think = None
    _qwen3_14b_awq_no_think = None
    _qwen3_32b_think = None

    @property
    def qwen3_14b_awq_think(self)->BaseChatOpenAI:
        if self._qwen3_14b_awq_think is None:
            self._qwen3_14b_awq_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                api_base="http://127.0.0.1:8000/v1",
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return self._qwen3_14b_awq_think

    @property
    def qwen3_14b_awq_no_think(self)->BaseChatOpenAI:
        if self._qwen3_14b_awq_no_think is None:
            self._qwen3_14b_awq_no_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                api_base="http://127.0.0.1:8000/v1",
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        return self._qwen3_14b_awq_no_think

    @property
    def qwen3_32b_think(self)->BaseChatOpenAI:
        if self._qwen3_32b_think is None:
            self._qwen3_32b_think = ChatDeepSeek(
                model="Qwen3-32B",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                api_base=os.environ.get("QWEN3_32B_BASE_URL", ""),
                max_tokens=1000,
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return self._qwen3_32b_think

if __name__ == '__main__':
    llm = LocalLLM()
    os.environ["QWEN3_32B_BASE_URL"]='https://iay-2dqexvlx16jnjgbdt-ldce0e7e-custom.service.onethingrobot.com/v1'
    llm.qwen3_32b_think.invoke('hello')