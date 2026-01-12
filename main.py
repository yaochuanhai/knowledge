import os
import json
from openai import OpenAI
import streamlit as st
from InterviewReport import InterviewReport

# 1. 页面配置
st.set_page_config(page_title="Java 毒舌面试官", page_icon="🤖")
st.title("🤖 尖酸刻薄的 Java 面试官")

# 加载本地环境变量（本地开发用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 优先读取 Streamlit Secrets
api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
base_url = st.secrets.get("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

if not api_key:
    st.error("❌ 未找到 API Key，请在 Streamlit Cloud 的 Settings -> Secrets 中配置")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)

# 2. 初始化 Session State (聊天记忆库)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个说话尖酸刻薄的面试官，专门挑 Java 程序员的刺。"}
    ]

# 3. 核心功能函数
def query_knowledge_base(topic: str):
    try:
        file_path = os.path.join(os.getcwd(), "knowledge.json")
        if not os.path.exists(file_path):
            return "知识库文件不存在，请检查 GitHub 仓库。"
        with open(file_path, "r", encoding="utf-8") as f: # 修正模式为 'r'
            kb = json.load(f)
        for key in kb:
            if topic.strip().lower() in key.lower():
                return kb[key]
        return "这种基础都不会？知识库里都没记这种破东西。"
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
    
    # 【关键修复】：清洗历史记录
    # 过滤掉包含 tool_calls 的消息和 role 为 tool 的消息
    # 只保留纯文本的对话内容，避免 API 校验序列失败
    clean_history = []
    for m in history:
        # 只保留有内容且不是工具调用中间态的消息
        if m["role"] in ["user", "system"]:
            clean_history.append(m)
        elif m["role"] == "assistant" and m.get("content"):
            # 排除掉只有 tool_calls 没有 content 的 assistant 消息
            clean_history.append({"role": "assistant", "content": m["content"]})

    messages = clean_history + [{"role": "system", "content": system_instruction}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False
    )
    raw_json_str = response.choices[0].message.content
    return InterviewReport.model_validate_json(raw_json_str)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_knowledge",
            "description": "当用户提到具体技术名词（如 HashMap, Spring）时，查阅内部知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "技术关键字"}
                },
                "required": ["topic"]
            }
        }
    }
]

# 4. 侧边栏：报告展示
with st.sidebar:
    st.header("面试设置")
    if st.button("🏁 结束面试并生成报告"):
        if len(st.session_state.messages) < 3:
            st.warning("话都没说两句就想跑？")
        else:
            with st.spinner("正在评估你的简历（废纸）..."):
                try:
                    report = get_final_report(st.session_state.messages)
                    st.session_state.report = report
                except Exception as e:
                    st.error(f"生成失败: {e}")

    if "report" in st.session_state:
        st.divider()
        r = st.session_state.report
        if r.is_hired:
            st.success(f"通过 (得分: {r.final_score})")
            st.balloons()
        else:
            st.error(f"淘汰 (得分: {r.final_score})")
        st.json(r.model_dump())

# 5. 渲染聊天历史
for message in st.session_state.messages:
    # 只要 role 不是 system 就显示
    if message["role"] != "system":
        # 兼容处理：tool 消息不直接展示在气泡中，或者可以自定义展示
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 6. 用户输入与 AI 逻辑
if prompt := st.chat_input("输入你的回答..."):
    # 记录并显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 回复逻辑
    with st.chat_message("assistant"):
        with st.spinner("面试官正在酝酿毒液..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=st.session_state.messages,
                tools=tools
            )
            
            msg_obj = response.choices[0].message
            
            # 关键修复：如果触发了工具调用
            if msg_obj.tool_calls:
                # 将消息对象转为字典后存入历史
                st.session_state.messages.append(msg_obj.model_dump())
                
                for tool_call in msg_obj.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if func_name == "get_knowledge":
                        res_content = query_knowledge_base(args.get('topic'))
                        # 记录工具返回结果
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(res_content)
                        })
                
                # 再次请求 AI 整合知识库内容进行回复
                second_res = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages
                )
                final_text = second_res.choices[0].message.content
            else:
                final_text = msg_obj.content
            
            # 显示并保存 AI 的最终回答
            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})