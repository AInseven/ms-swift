from prompt import  McpExtractorModel, mcp_extractor_prompt,mcp_rating_prompt,McpScore
from typing import Union, Literal, List
from structured_llm_output import prompt_parsing
import time
from local_vllm import LocalLLM

llm = LocalLLM()

def batch_extract_answer(llm_response_list: list[str]) -> Union[
    Literal["A", "B", "C", "D"],
    List[Literal["A", "B", "C", "D", "E"]],
]:
    """
    Input: List[str]
    Output: List[Answer] (either str or list[str])
    """
    input_query = [mcp_extractor_prompt.format(answer=res) for res in llm_response_list]
    failed_model = McpExtractorModel(answer="no_answer")
    extractor_model_list = prompt_parsing(
        model=McpExtractorModel,
        failed_model=failed_model,
        query=input_query,
        llm=llm.qwen3_32b_think,
        langfuse_user_id='extract_answer',
        langfuse_session_id=time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )

    return [each.answer for each in extractor_model_list]


def batch_rate_mcp(llm_response_list: list[str], ground_truth_list: list[str]) -> List[float]:
    input_query = [mcp_rating_prompt.format(correct_ans=gt, answer=res) for gt, res in zip(ground_truth_list, llm_response_list)]
    failed_model = McpScore(score=0)
    res = prompt_parsing(
        model=McpScore,
        failed_model=failed_model,
        query=input_query,
        llm=llm.qwen3_32b_fp8_think,
        langfuse_user_id='batch_rate_mcp',
        langfuse_session_id=time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )
    return [each.score for each in res]

