import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 解决 HuggingFace 下载网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

# --- 页面基础设置 ---
st.set_page_config(page_title="专属 AI 知识库", page_icon="📚", layout="wide")
st.title("📚 专属 AI 私有知识库")
st.markdown("上传你的 PDF 文件，让 AI 帮你阅读并回答问题。")

# --- 初始化模型 ---
@st.cache_resource # 这个装饰器让模型只加载一次，避免网页刷新时卡顿
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )

llm = get_llm()

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("请上传你的 PDF 文件", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("文件已上传！")

# --- 核心逻辑：处理文档并建立知识库 ---
@st.cache_resource(show_spinner="正在解析文档并建立知识库，请稍候...")
def process_pdf(file_path):
    # 1. 加载 PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    if not docs or len(docs[0].page_content) == 0:
        return None
    
    # 2. 切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # 3. 向量化并存入数据库
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# 如果上传了文件，就进行处理
if uploaded_file is not None:
    # 将上传的文件保存到本地临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    vectorstore = process_pdf(tmp_file_path)
    
    if vectorstore is None:
        st.error("无法从该 PDF 中读取文本，请检查是否是扫描版图片。")
    else:
        # --- 建立 RAG 对话链 ---
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        system_prompt = (
            "你是一个专业的文档问答助手。请严格根据以下提供的上下文来回答问题。"
            "如果你不知道答案，就说你不知道。输出纯文本，不要使用 Markdown 符号。\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # --- 聊天界面 ---
        # 初始化聊天记录
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 展示历史聊天记录
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 接收用户输入
        if user_input := st.chat_input("针对这份文档，你想问什么？"):
            # 显示用户提问
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 调用 AI 回答
            with st.chat_message("assistant"):
                with st.spinner("AI 正在查阅文档..."):
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
            
            # 保存 AI 回答
            st.session_state.messages.append({"role": "assistant", "content": answer})