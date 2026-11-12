import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, create_retriever_tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
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
    .ad-banner { background: linear-gradient(90deg, #3b82f6 0%, #2dd4bf 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ad-banner">
    <div class="ad-title" style="font-size:24px; font-weight:bold; margin-bottom:10px;">🛍️ 智选优品 · 堡垒级双引擎客服</div>
    <div class="ad-subtitle">类目边界拦截 | 金牌导购追问 | 绝对零幻觉 | 连续2次情绪失控自动转人工</div>
</div>
""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 数据看板", "📦 商品管理", "💬 全能 AI 客服（情绪识别版）"])

with tab1:
    st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['访客数', '转化率']))

with tab2:
    st.dataframe(pd.DataFrame({"商品": ["纯棉短袖", "冰丝防晒衣", "雪纺V领衬衫"], "价格": [99, 159, 129]}), use_container_width=True)
    # ==========================================
# 2. 核心：堡垒级双引擎 Agent
# ==========================================
with tab3:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_negative_count" not in st.session_state:
        st.session_state.total_negative_count = 0
    if "consecutive_negative_count" not in st.session_state:
        st.session_state.consecutive_negative_count = 0

    @st.cache_resource 
    def get_llm():
        return ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com", temperature=0.0) 
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
        """当用户发送商品链接（URL）时，必须立刻使用此工具获取该商品信息。"""
        if "111" in url: return "【已锁定商品】：纯棉短袖。价格99元。材质：100%新疆棉。现有尺码：M, L, XL。无其他尺码。"
        elif "222" in url: return "【已锁定商品】：冰丝防晒衣。价格159元。材质：UPF50+。现有尺码：均码。"
        return "【已锁定商品】：基础款服饰。"

    @tool
    def get_current_promotions() -> str:
        """当用户询问优惠、活动、满减、包邮时，调用此工具。"""
        return "当前店铺活动：全场满99元包邮。暂未开启关注店铺领券活动。无其他满减。"

    @tool
    def analyze_product_image(image_name: str) -> str:
        """当用户发送图片时，调用此工具识别图片特征。"""
        if "短袖" in image_name: return "【识图结果】：黑色、纯棉材质的圆领短袖。"
        elif "别家" in image_name: return "【识图结果】：白色、真丝材质的V领衬衫。"
        return "【识图结果】：识别到一张服装图片。"

    @tool
    def search_similar_products(features: str) -> str:
        """根据特征（如颜色、材质）搜索相似的商品库。注意：店铺只卖服装类，没有手机、家具等其他商品。"""
        db = [
            {"name": "纯棉短袖", "color": "白色, 黑色", "material": "100%纯棉", "price": 99, "desc": "基础款圆领短袖。"},
            {"name": "冰丝防晒衣", "color": "白色, 粉色", "material": "88%锦纶", "price": 159, "desc": "带连帽，UPF50+。"},
            {"name": "雪纺V领衬衫", "color": "白色, 杏色", "material": "100%聚酯纤维(仿丝绸)", "price": 129, "desc": "V领显瘦，可作真丝平替。"}
        ]
        matches = []
        for item in db:
            if ("短袖" in features and "短袖" in item["name"]) or ("衬衫" in features and "衬衫" in item["name"]) or ("黑" in features and "黑" in item["color"]) or ("真丝" in features and "仿丝绸" in item["material"]):
                matches.append(f"商品：{item['name']}（{item['color']}），材质：{item['material']}，价格：{item['price']}元。")
        return "\n".join(matches) if matches else "商品库中暂无匹配的商品。"

    @tool
    def check_order_status(order_id: str) -> str:
        """售后工具：查询订单状态和物流轨迹。"""
        return "订单已发货，目前在深圳转运中心。"

    @tool
    def get_after_sales_policy(topic: str) -> str:
        """售后工具：查询退换货、退款等店规。"""
        return "店规：支持7天无理由退换（吊牌完好），有运费险可抵扣。"

    tools = [get_product_info_by_url, get_current_promotions, analyze_product_image, search_similar_products, check_order_status, get_after_sales_policy]

    chat_col1, chat_col2 = st.columns([1, 3])
    with chat_col1:
        st.markdown("### 📚 知识库与发图")
        uploaded_files = st.file_uploader("上传店规/尺码表", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        buyer_image = st.file_uploader("模拟买家发图", type=["jpg", "png", "jpeg"])
        if buyer_image:
            if st.button("发送图片给客服"):
                st.session_state.pending_image = buyer_image.name

        if st.button("🧹 清空对话与情绪", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_negative_count = 0
            st.session_state.consecutive_negative_count = 0
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
                retriever_tool = create_retriever_tool(retriever, "knowledge_base_search", "查询店铺的店规、尺码表等静态知识。")

        all_tools = tools.copy()
        if retriever_tool:
            all_tools.append(retriever_tool)

        system_prompt = """你是「智选优品」服装店的专属高级客服。

【全局最高纪律】
1. 语气像真人，简洁直接。绝对不能使用“你”，**必须全部使用“您”**来称呼买家。
2. **【门店类目护城河（极重要）】**：本店是**服装店**，只卖短袖、防晒衣、衬衫。如果买家让你推荐“手机、办公椅、沙发、情趣用品”等**非服装类商品**，**绝对不允许顺着往下问**，必须立刻拦截并引导回服装：“抱歉，小智这里是服装店，目前只卖衣服哦。请问您需要看看短袖吗？”
3. **【绝对禁止常识脑补（零幻觉）】**：你只能根据“工具返回的数据”或“知识库内容”回答。没写 XXXL 码就直接说没有，禁止推测。
4. 遇到无法处理的售后问题，必须输出指定的JSON格式转人工。

【场景一：售前导购（金牌销冠逻辑）】
- **【模糊需求必须追问】**：当买家说“推荐一款短袖/衣服”等没有具体明确要求的话时，不允许直接罗列商品。必须先反问：“好的，为了给您推荐最合适的，请问是男生还是女生穿？平时喜欢什么颜色，大概想要什么尺码呢？”
- **【精准需求直接推荐】**：如果买家说“推荐一款黑色的纯棉L码短袖”，需求明确，直接调用 `search_similar_products` 推荐。
- **【尺码前置强约束】**：买家没发链接直接问尺码，必须反问：“请问您看中的是哪款商品呢？”
- **【链接必查】**：收到链接，必须调用 `get_product_info_by_url`。
- **【优惠必查】**：问到优惠，必须调用 `get_current_promotions`。
- 图片：必须先调 `analyze_product_image`，再调 `search_similar_products`。

【场景二：售后客服】
- 涉及退款金额、赔付，绝对不能自行裁定！必须输出JSON转人工。

【转人工输出格式】
{{"action":"handoff","reason":"写原因","user_facing":"写安抚用户并告知转接的话"}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, all_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True)

        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        user_input = st.chat_input("在这里输入您的需求...")
        if "pending_image" in st.session_state and st.session_state.pending_image:
            user_input = f"[买家发送了一张图片：{st.session_state.pending_image}] 你们家有这件吗？"
            st.session_state.pending_image = None 

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            force_handoff = False
            handoff_reason = ""
            user_facing_msg = ""

            sensitive_words = ["转人工", "投诉", "12315", "赔偿", "退钱", "赔钱", "你们领导"]
            if any(word in user_input for word in sensitive_words):
                force_handoff = True
                handoff_reason = "触发前置敏感词规则"

            negative_words = ["太差", "垃圾", "无语", "气死", "骗子", "恶心", "差评", "问问问", "问那么多", "烦不烦", "别问了", "智障", "傻", "蠢", "神经病", "有病", "滚"]
            is_negative_now = any(word in user_input for word in negative_words)
            
            if is_negative_now:
                st.session_state.total_negative_count += 1
                st.session_state.consecutive_negative_count += 1
            else:
                st.session_state.consecutive_negative_count = 0
                
            if st.session_state.total_negative_count >= 3:
                force_handoff = True
                handoff_reason = "累计负面情绪达到3次，强制介入"
            elif st.session_state.consecutive_negative_count >= 2:
                force_handoff = True
                handoff_reason = "连续2次出现负面情绪，强制介入"
            elif is_negative_now:
                user_facing_msg = "非常抱歉让您心烦了。小智多问几句，主要是为了更准确地了解您的需求，好给您推荐最合适的。如果您不想挑选，我直接给您推荐店里卖得最好的爆款纯棉短袖（99元），您看行吗？"
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(user_facing_msg)
                        st.warning(f"⚠️ **预警**：检测到买家情绪不佳（当前连续：{st.session_state.consecutive_negative_count}/2，累计：{st.session_state.total_negative_count}/3）。")
                st.session_state.messages.append({"role": "assistant", "content": user_facing_msg})
                st.stop()

            if force_handoff:
                user_facing_msg = "非常抱歉给您带来了不好的体验，我马上为您转接人工专员，请稍等。（人工时段：9:00-22:00）"
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(user_facing_msg)
                        st.error(f"🚨 **前置风控拦截**：{handoff_reason}")
                st.session_state.messages.append({"role": "assistant", "content": user_facing_msg})
                st.stop()

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("客服正在处理..."):
                        chat_history = []
                        for msg in st.session_state.messages[:-1]:
                            if msg["role"] == "user": chat_history.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant": chat_history.append(AIMessage(content=msg["content"]))
                        
                        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
                        answer = response["output"]

                        display_answer = answer
                        try:
                            json_match = re.search(r'\{.*"action"\s*:\s*"handoff".*\}', answer, re.DOTALL)
                            if json_match:
                                data = json.loads(json_match.group(0))
                                if data.get("action") == "handoff":
                                    reason = data.get("reason", "模型越界拦截")
                                    user_facing = data.get("user_facing", "抱歉，正在为您转接人工。")
                                    display_answer = user_facing
                                    st.markdown(user_facing)
                                    st.error(f"🚨 **模型安全拦截**：已转人工。内部原因：{reason}")
                                else:
                                    st.markdown(answer)
                            else:
                                st.markdown(answer)
                        except Exception:
                            st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": display_answer})