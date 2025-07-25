from pydantic import BaseModel, Field
from typing import List,Dict,Literal,Union
from langchain_core.output_parsers import PydanticOutputParser

mcp_rating_prompt="""
每道题包含一个“正确答案”和一个“回答”。回答中可能包含解释或分析，请你提取其中的最终答案部分，并按以下规则打分：

若为单选题，则当“回答中的最终答案”与“正确答案”完全一致时，得 1 分，否则得 0 分；
若为多选题，则只有在“回答中的所有选项”与“正确答案”在内容上**完全一致（顺序无关）**时得 1 分，否则得 0 分；
若回答中没有明确提及选项的编号ABCD，只有选项的内容，得0分。

**正确答案：** `{correct_ans}`
**回答：** `{answer}`
"""

class McpScore(BaseModel):
    # Literal 只能是0或1
    score: Literal[0, 1] = Field(..., description="The score of the answer, only 0 or 1")

mcp_rating_parser=PydanticOutputParser(pydantic_object=McpScore)

def mcp_rate_msg(correct_ans,answer)->List[Dict]:
    return [
        {"role": "system", "content": f"你是一个选择题自动评分器。请按要求打分，并按照如下要求返回:\n{mcp_rating_parser.get_format_instructions()}"},
        {"role": "user", "content": mcp_rating_prompt.format(correct_ans=correct_ans, answer=answer)},
    ]


class McpExtractorModel(BaseModel):
    answer: Union[
        Literal["A", "B", "C", "D"],                   # single choice
        List[Literal["A", "B", "C", "D", "E"]],         # multiple choice
        Literal["no_answer"]
    ] = Field(..., description="提取的答案。如果是单选题，就返回str；如果是多选题，就返回list。如果回答中没有答案，就返回no_answer")


mcp_extractor_prompt= """
你的任务是从回答中提取最终选择的答案。回答可能包含分析、推理或多个阶段的思考，但你只需要返回最终的选择结果，不包含多余内容。

回答：
{answer}
"""