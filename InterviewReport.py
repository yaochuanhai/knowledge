from pydantic import BaseModel
from typing import List

class InterviewReport(BaseModel):
    candidate_name: str
    final_score: int          # 0-100
    top_3_weaknesses: List[str] # 候选人的三个致命弱点
    is_hired: bool            # 是否录用
    sharp_summary: str        # 一句刻薄的总评