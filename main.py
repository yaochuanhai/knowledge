import os
import asyncio
import json
from openai import OpenAI
from dotenv import load_dotenv
from InterviewReport import InterviewReport
import streamlit as st

# 尝试加载本地 .env，如果失败（比如在云端）也不报错
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 优先从 Streamlit Secrets 读取，其次从环境变量读取
api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
base_url = st.secrets.get("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)
history_path = os.path.join(os.getcwd(),"history.json")

def load_history():
    if os.path.exists(history_path):
        with open(history_path,"r",encoding="utf-8") as f:
             json.load(f)
    else:
        return [{"role":"system","content": "你是一个说话尖酸刻薄的面试官，专门挑 Java 程序员的刺。"}]

def save_history(history):
    with open(history_path,"w",encoding="utf-8") as f:
        json.dump(history,f,ensure_ascii=False,indent=4)



history = load_history()

def query_knowledge_base(topic: str):
    try:
        file_path = os.path.join(os.getcwd(),"knowledge.json")
        with open(file_path,"r",encoding="utf-8") as f:
            kb = json.load(f)
        for key in kb:
            if topic.strip().lower() in key:
                return kb[key]

    except Exception as e:
        return f"查阅异常{e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_knowledge",
            "description": "当用户回答HashMap、Spring时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string", 
                        "description": "技术栈"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

def get_final_report(history):
    # 构造一个“总结指令”
    system_instruction = (
        "你是一个严格的 JSON 生成器。请根据对话总结面试报告。"
        "必须输出以下结构的 JSON，不得包含任何其他文字：\n"
        "{\n"
        '  "candidate_name": "姓名",\n'
        '  "final_score": 80,\n'
        '  "top_3_weaknesses": ["弱点1", "弱点2", "弱点3"],\n'
        '  "is_hired": true,\n'
        '  "sharp_summary": "刻薄评语"\n'
        "}"
    )
    messages= history + [{"role":"system","content":system_instruction}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False
    )
    raw_json_str = response.choices[0].message.content
    report_data = InterviewReport.model_validate_json(raw_json_str)
    return report_data


async def chat_with_tools(user_input):
    history.append({"role":"user","content":user_input})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        tools=tools
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        history.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[系统日志] 🔍 AI 正在后台查阅秘籍...")

            if func_name == "get_knowledge":
                res_content = query_knowledge_base(args.get('topic'))
            history.append({"role":"tool","tool_call_id": tool_call.id,"content":res_content})

        second_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history
        )
        final_msg = second_response.choices[0].message.content
        print(f"🤖 AI: {final_msg}")
        history.append({"role": "assistant", "content": str(final_msg)})
    else:
        print(f"🤖 AI: {msg.content}")
        history.append({"role": "assistant", "content": msg.content})



async def main():
        print("🤖 面试官提醒 (输入 'finalize' 结束面试)")
        while True:
            query = input("\n👨‍💻 你: ")
            if query.lower() == 'finalize':
                res = get_final_report(history)
                file_path = os.path.join(os.getcwd(), "report.json")

                with open(file_path,"w",encoding="utf-8") as f:
                    json.dump(res.model_dump(),f,ensure_ascii=False,indent=4)
                if res.is_hired:
                    print("算你走运，明天来上班。")
                else:
                    print("果然不出所料，你可以滚了。")
                break
            await chat_with_tools(query)
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e :
        print(f"发生致命错误: {e}")