import streamlit as st
import requests

def main():
    st.title("MiaOllama 测试页面")
    
    # 测试Ollama连接
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            st.success("成功连接到Ollama服务！")
            models = response.json().get("models", [])
            st.write("可用模型：", [model["name"] for model in models])
        else:
            st.error("无法连接到Ollama服务")
    except Exception as e:
        st.error(f"连接错误：{str(e)}")
    
    # 简单的聊天界面
    with st.form("chat_form"):
        user_input = st.text_input("输入消息：")
        model = st.selectbox("选择模型", ["llama2", "mistral", "gemma"])
        submitted = st.form_submit_button("发送")
        
        if submitted and user_input:
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": user_input}
                        ]
                    }
                )
                if response.status_code == 200:
                    st.write("助手：", response.json()["message"]["content"])
                else:
                    st.error("请求失败")
            except Exception as e:
                st.error(f"错误：{str(e)}")

if __name__ == "__main__":
    main()
