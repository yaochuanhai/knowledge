import os
import json
from openai import OpenAI
import streamlit as st
from InterviewReport import InterviewReport

# 1. 页面配置与环境初始化
st.set_page_config(page_title="Java 毒舌面试官", page_icon="🤖")
st.title("🤖 尖酸刻薄的 Java 面试官")

# 加载本地环境变量（仅限本地开发）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 优先读取 Streamlit Secrets
api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
base_url = st.secrets.get("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

if not api_key:
    st.error("未找到 API Key，请在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)

# 2. 初始化 Session State (替代 history.json)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个说话尖酸刻薄的面试官，专门挑 Java 程序员的刺。"}
    ]

# 3. 辅助函数
def query_knowledge_base(topic: str):
    try:
        # 注意：确保 knowledge.json 已经上传到 GitHub 仓库根目录
        file_path = os.path.join(os.getcwd(), "knowledge.json")
        if not os.path.exists(file_path):
            return "知识库文件缺失。"
        with open(file_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        for key in kb:
            if topic.strip().lower() in key.lower():
                return kb[key]
        return "未找到相关技术知识。"
    except Exception as e:
        return f"查阅异常: {e}"

def get_final_report(history):
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
    messages = history + [{"role": "system", "content": system_instruction}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False
    )
    raw_json_str = response.choices[0].message.content
    return InterviewReport.model_validate_json(raw_json_str)

# 函数调用工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_knowledge",
            "description": "当用户回答HashMap、Spring等技术栈时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "技术栈名称"}
                },
                "required": ["topic"]
            }
        }
    }
]

# 4. 侧边栏：功能控制
with st.sidebar:
    st.header("面试控制台")
    if st.button("🏁 结束面试并生成报告"):
        if len(st.session_state.messages) < 3:
            st.warning("面试还没开始呢，急着投胎吗？")
        else:
            with st.spinner("正在生成毒舌报告..."):
                try:
                    report = get_final_report(st.session_state.messages)
                    st.session_state.report = report
                except Exception as e:
                    st.error(f"生成报告失败: {e}")

    if "report" in st.session_state:
        st.divider()
        res = st.session_state.report
        if res.is_hired:
            st.success("算你走运，明天来上班。")
        else:
            st.error("果然不出所料，你可以滚了。")
        st.json(res.model_dump())

# 5. 聊天界面渲染
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 6. 用户输入处理
if prompt := st.chat_input("说点什么来取悦面试官..."):
    # 立即展示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 AI 回复
    with st.chat_message("assistant"):
        with st.spinner("面试官正在酝酿毒液..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=st.session_state.messages,
                tools=tools
            )
            
            msg = response.choices[0].message
            
            # 处理 Tool Calls (Function Calling)
            if msg.tool_calls:
                st.session_state.messages.append(msg)
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if func_name == "get_knowledge":
                        res_content = query_knowledge_base(args.get('topic'))
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": res_content
                        })
                
                # 第二次请求，获取最终文字回复
                second_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages
                )
                final_content = second_response.choices[0].message.content
            else:
                final_content = msg.content
            
            st.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})