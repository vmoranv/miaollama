import streamlit as st
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import datetime

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.prompt.advanced_prompt_manager import AdvancedPromptManager
from src.core.llm.ollama_client import OllamaClient
import json

def init_session_state():
    """初始化会话状态"""
    if "prompt_manager" not in st.session_state:
        st.session_state.prompt_manager = AdvancedPromptManager()
    
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama2"
    
    if "editing_template" not in st.session_state:
        st.session_state.editing_template = None

def create_prompt_page():
    """创建提示词编辑页面"""
    st.title("提示词编辑")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["创建提示词", "管理提示词", "优化提示词"])
    
    # 创建提示词标签页
    with tab1:
        with st.form("create_form"):
            st.subheader("创建新提示词")
            name = st.text_input("名称")
            description = st.text_input("描述")
            content = st.text_area("内容", height=300)
            
            submitted = st.form_submit_button("保存")
            if submitted:
                if name and content:
                    template = {
                        "name": name,
                        "description": description,
                        "content": content
                    }
                    if st.session_state.prompt_manager.save_template(template):
                        st.success("提示词已保存")
                        st.rerun()
                    else:
                        st.error("保存失败")
                else:
                    st.error("名称和内容不能为空")
    
    # 管理提示词标签页
    with tab2:
        st.subheader("管理提示词")
        templates = st.session_state.prompt_manager.list_templates()
        if templates:
            for i, template in enumerate(templates):
                with st.expander(f"{template['name']} - {template['description'] or '无描述'}"):
                    st.text_area(
                        "内容",
                        template["content"],
                        height=200,
                        disabled=True
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("编辑", key=f"edit_{i}"):
                            st.session_state.editing_template = template
                            st.rerun()
                    with col2:
                        if st.button("删除", key=f"delete_{i}"):
                            if st.session_state.prompt_manager.delete_template(template["name"]):
                                st.success("提示词已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")
        else:
            st.info("暂无提示词模板")
        
        # 编辑模板对话框
        if st.session_state.editing_template:
            with st.form("edit_form"):
                st.subheader("编辑提示词")
                name = st.text_input(
                    "名称",
                    st.session_state.editing_template.get("name", "")
                )
                description = st.text_input(
                    "描述",
                    st.session_state.editing_template.get("description", "")
                )
                content = st.text_area(
                    "内容",
                    st.session_state.editing_template.get("content", ""),
                    height=300
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("保存"):
                        if name and content:
                            template = {
                                "name": name,
                                "description": description,
                                "content": content
                            }
                            if st.session_state.prompt_manager.save_template(template):
                                st.session_state.editing_template = None
                                st.success("提示词已更新")
                                st.rerun()
                            else:
                                st.error("更新失败")
                        else:
                            st.error("名称和内容不能为空")
                
                with col2:
                    if st.form_submit_button("取消"):
                        st.session_state.editing_template = None
                        st.rerun()
    
    # 优化提示词标签页
    with tab3:
        def optimize_prompt_page():
            """优化提示词页面"""
            st.header("优化提示词")
            
            # 获取本地模型列表
            client = OllamaClient()
            models = client.get_models()
            
            # 输入提示词
            prompt = st.text_area("输入需要优化的提示词", height=200)
            
            # 选择模型
            model = st.selectbox("选择模型", models, index=models.index("llama2") if "llama2" in models else 0)
            
            if st.button("开始优化"):
                if not prompt:
                    st.error("请输入需要优化的提示词")
                    return
                
                with st.spinner("正在优化提示词..."):
                    try:
                        # 创建提示词管理器实例
                        prompt_manager = AdvancedPromptManager()
                        
                        # 调用优化函数
                        result = prompt_manager.optimize_prompt(prompt, model)
                        
                        if isinstance(result, dict):
                            if "error" in result:
                                st.error(f"优化失败：{result['error']}")
                            else:
                                # 显示分析结果
                                st.subheader("分析结果")
                                st.write(result.get("分析", ""))
                                
                                st.subheader("优化建议")
                                st.write(result.get("优化建议", ""))
                                
                                st.subheader("优化后的提示词")
                                optimized_prompt = result.get("优化后的提示词", "")
                                st.text_area("优化后的提示词", value=optimized_prompt, height=200)
                                
                                # 添加保存按钮
                                if st.button("保存为新模板"):
                                    try:
                                        # 创建新模板
                                        template = {
                                            "name": f"优化模板_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                            "content": optimized_prompt,
                                            "description": "通过AI优化生成的提示词模板",
                                            "category": "优化模板",
                                            "tags": ["优化", "AI生成"],
                                            "variables": [],
                                            "author": "AI优化器",
                                            "version": "1.0",
                                            "created_at": datetime.datetime.now().isoformat(),
                                            "updated_at": datetime.datetime.now().isoformat()
                                        }
                                        
                                        # 保存模板
                                        template_id = prompt_manager.save_template(template)
                                        if template_id:
                                            st.success("模板保存成功！")
                                        else:
                                            st.error("模板保存失败")
                                    except Exception as e:
                                        st.error(f"保存模板时出错：{str(e)}")
                        else:
                            st.error("优化结果格式错误")
                    except Exception as e:
                        st.error(f"优化过程中出错：{str(e)}")
        
        optimize_prompt_page()

def chat_page():
    """聊天界面"""
    st.title("聊天界面")
    
    # 侧边栏 - 提示词模板选择
    with st.sidebar:
        st.subheader("提示词模板")
        templates = st.session_state.prompt_manager.list_templates()
        if templates:
            selected_template = st.selectbox(
                "选择提示词模板",
                options=[t["name"] for t in templates],
                format_func=lambda x: x,
                key="template_selector"
            )
            if st.button("应用模板", key="apply_template"):
                template = next((t for t in templates if t["name"] == selected_template), None)
                if template:
                    # 保留用户消息，只替换系统消息
                    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
                    st.session_state.messages = [{"role": "system", "content": template["content"]}] + user_messages
                    st.success("已应用提示词模板")
                    st.rerun()
    
    # 主界面
    handle_chat()

def handle_chat():
    """处理聊天功能"""
    # 显示聊天记录
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)
    
    # 获取用户输入
    if prompt := st.chat_input("输入你的消息..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # 添加助手消息占位
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # 获取回复
                for response in st.session_state.ollama_client.chat_stream(
                    st.session_state.messages,
                    model=st.session_state.current_model
                ):
                    if response:
                        full_response += response
                        # 使用div包装整个响应，确保样式一致
                        formatted_response = f'<div class="chat-response">{full_response}</div>'
                        message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                
                # 更新消息历史
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        except Exception as e:
            st.error(f"生成回答时出错：{str(e)}")

def init():
    """初始化应用"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama2"
    
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
    
    if "prompt_manager" not in st.session_state:
        st.session_state.prompt_manager = AdvancedPromptManager()
    
    if "editing_template" not in st.session_state:
        st.session_state.editing_template = None
    
    # 设置页面配置，允许HTML和unsafe HTML
    st.set_page_config(
        page_title="聊天界面",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 注入CSS样式
    st.markdown("""
        <style>
        .chat-response pre {
            color: #666666 !important;
            border-left: 3px solid #666666 !important;
            padding-left: 1em !important;
            margin: 1em 0 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            background-color: transparent !important;
            border: none !important;
            font-family: inherit !important;
            font-size: inherit !important;
            line-height: 1.5 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    init()
    init_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.title("设置")
        
        # 模型选择
        available_models = st.session_state.ollama_client.get_models()
        current_index = available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
        st.session_state.current_model = st.selectbox(
            "选择模型",
            available_models,
            index=current_index,
            key="model_selector"
        )
        
        # 添加分隔线
        st.markdown("---")
        
        # 页面选择（使用session_state存储当前页面）
        if "current_page" not in st.session_state:
            st.session_state.current_page = "聊天界面"
        
        # 使用单一的页面选择器
        st.session_state.current_page = st.radio(
            "选择页面",
            ["聊天界面", "提示词编辑"],
            key="main_page_selection"
        )
    
    # 主界面
    if st.session_state.current_page == "聊天界面":
        chat_page()
    else:
        create_prompt_page()

if __name__ == "__main__":
    main()
