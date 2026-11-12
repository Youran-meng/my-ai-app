import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, create_retriever_tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
# 引入消息类型
from langchain_core.messages import HumanMessage, AIMessage

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

try:
    if "DEEPSEEK_API_KEY" in st.secrets:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")
except Exception:
    api_key = os.getenv("DEEPSEEK_API_KEY")

# ==========================================
# 1. 网页外观与指标
# ==========================================
st.set_page_config(page_title="智选 AI 商家中台", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .ad-banner { background: linear-gradient(90deg, #ff4b4b 0%, #ff7e5f 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ad-banner">
    <div class="ad-title" style="font-size:24px; font-weight:bold; margin-bottom:10px;">🏆 拒绝废话，绝不瞎编！电商老板的专属 AI 客服</div>
    <div class="ad-subtitle">多模态链接解析 | 多轮对话记忆 | 实时库存物流 | 防幻觉兜底</div>
</div>
""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 数据看板", "📦 库存管理", "💬 AI 客服中心（Agent 记忆版）"])

with tab1:
    st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['销售额', '访客数']))

with tab2:
    st.dataframe(pd.DataFrame({"商品": ["冰丝防晒衣", "纯棉短袖"], "M码库存": [50, 100], "L码库存": [10, 80], "XL码库存": [0, 20]}), use_container_width=True)

# ==========================================
# 2. 核心：Agent 客服中心
# ==========================================
with tab3:
    @st.cache_resource 
    def get_llm():
        return ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com", temperature=0.1)
    llm = get_llm()

    @st.cache_resource(show_spinner="正在载入店铺知识库...")
    def process_files(file_paths_tuple):
        all_splits = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        for file_path in file_paths_tuple:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf": loader = PyPDFLoader(file_path)
                elif ext == ".docx": loader = Docx2txtLoader(file_path)
                elif ext == ".txt": loader = TextLoader(file_path, encoding="utf-8")
                else: continue
                docs = loader.load()
                if docs and len(docs[0].page_content) > 0:
                    all_splits.extend(text_splitter.split_documents(docs))
            except Exception: pass
        if not all_splits: return None
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        return Chroma.from_documents(documents=all_splits, embedding=embeddings)

    @tool
    def get_product_info_by_url(url: str) -> str:
        """当用户发送一个商品链接（URL）时，必须立刻使用此工具获取该商品的详细信息。"""
        if "111" in url:
            return "【已锁定商品】：纯棉短袖。材质：100%新疆棉，透气吸汗。支持七天无理由退换。"
        elif "222" in url:
            return "【已锁定商品】：冰丝防晒衣。材质：88%锦纶+12%氨纶，UPF50+。不可机洗，不支持七天无理由。"
        else:
            return "【已锁定商品】：商品信息解析成功，这是一件基础款服饰。"

    @tool
    def check_inventory(size: str) -> str:
        """当用户询问商品某个尺码是否有货、库存还剩多少时，使用此工具查询实时库存。参数 size 必须是 M, L, 或 XL。"""
        db = {"M": "50件", "L": "10件", "XL": "0件（已售罄）"}
        result = db.get(size.upper(), "未知尺码")
        return f"当前 {size.upper()} 码库存为：{result}"

    @tool
    def track_logistics(order_id: str) -> str:
        """当用户提供订单号，询问物流到哪了时，使用此工具。"""
        return f"订单 {order_id} 的最新状态：快件已到达【深圳南山转运中心】，预计今天下午送达。"

    tools = [get_product_info_by_url, check_inventory, track_logistics]

    chat_col1, chat_col2 = st.columns([1, 2])
    with chat_col1:
        st.markdown("### 📚 静态知识库")
        uploaded_files = st.file_uploader("上传通用售后规则(选填)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        st.info("💡 **多轮测试**：\n1. '我要查物流'\n2. '订单号 8888'\n(AI应该直接查，不再反问)")
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with chat_col2:
        retriever_tool = None
        if uploaded_files:
            temp_paths = []
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)
            
            vectorstore = process_files(tuple(temp_paths))
            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                retriever_tool = create_retriever_tool(
                    retriever, "knowledge_base_search", "当用户询问店铺通用的售后规则、发货政策时使用。"
                )

        all_tools = tools.copy()
        if retriever_tool:
            all_tools.append(retriever_tool)

        # 【优化】：强化多轮对话逻辑的提示词
        system_prompt = """你是店铺的王牌客服“小智”。
【核心纪律】：
1. 语气热情简短（50字以内），多用“亲”、“哈”。
2. **多轮对话记忆**：结合上下文理解。如果买家上一句说查物流，这一句发了纯数字，说明这就是订单号，**立刻调用【track_logistics】工具**，绝对不要反问！
3. 收到链接立刻调用【get_product_info_by_url】。
4. 问尺码库存调用【check_inventory】。
5. 问通用规则调用【knowledge_base_search】。
6. **【闲聊拦截】**：完全无关的问题，直接回复：“亲亲，小智只负责产品相关问题哦，有什么商品问题随时问我哈~”
7. **【业务兜底】**：查不到信息，回复：“亲，稍等，我帮您呼叫人工客服确认一下哈~”
"""
        # 【优化】：加入 chat_history 占位符
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # 这里是历史记忆
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, all_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container(height=450)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        user_input = st.chat_input("模拟买家提问...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("AI 正在思考..."):
                        # 【关键修复】：将 st.session_state.messages 转为 LangChain 认识的格式
                        chat_history = []
                        for msg in st.session_state.messages[:-1]: # 排除当前刚输入的这一句
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                chat_history.append(AIMessage(content=msg["content"]))
                        
                        # 将 chat_history 传给 Agent
                        response = agent_executor.invoke({
                            "input": user_input,
                            "chat_history": chat_history
                        })
                        answer = response["output"]
                        st.markdown(answer)
            
            if "人工客服" in answer:
                st.error("🚨 **触发安全网**：业务问题超出知识边界，已转人工！")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})