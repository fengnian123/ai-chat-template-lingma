import gradio as gr
import openai
import os
from functools import partial

# 魔搭社区API配置
API_BASE_URL = 'https://api-inference.modelscope.cn/v1/'
MODEL_ID = 'Qwen/Qwen3-235B-A22B'
API_KEY = '9b16f530-adcd-4711-a998-20e070d1265d'

# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# 自定义CSS样式
custom_css = """ .chatbot {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
.user {background-color: #DCF8C6; align-self: flex-end; border-radius: 8px 4px 8px 8px;}
.bot {background-color: #ECECEC; align-self: flex-start; border-radius: 4px 8px 8px 8px;}
.chat-message {padding: 0.5rem 1rem; margin: 0.5rem 0; font-size: 1.1rem;}
#chat-container {max-width: 800px; margin: 0 auto;}
@media (max-width: 600px) {.chat-message {font-size: 1rem;}} """


def format_response(response):
    """格式化API响应，分离思考过程和最终答案"""
    thinking_content = []
    answer_content = []
    done_thinking = False
    
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            thinking_content.append(chunk.choices[0].delta.reasoning_content)
        elif chunk.choices[0].delta.content:
            if not done_thinking:
                answer_content.append('\n\n=== Final Answer ===\n')
                done_thinking = True
            answer_content.append(chunk.choices[0].delta.content)
    
    return {
        'thinking': ''.join(thinking_content),
        'answer': ''.join(answer_content)
    }


def chatbot_response(message, chat_history, api_key=None):
    """处理用户消息并返回AI回复
    
    Args:
        message: 用户输入文本
        chat_history: 历史对话记录
        api_key: ModelScope API密钥（可选）
    
    Returns:
        AI回复内容
    """
    try:
        # 使用传入的API密钥或环境变量中的密钥
        effective_api_key = api_key if api_key else API_KEY
        
        # 构建API请求参数
        extra_body = {
            "enable_thinking": True,
            "thinking_budget": 4096
        }
        
        # 发送API请求
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                *[{'role': h['role'], 'content': h['content']} for h in chat_history],
                {'role': 'user', 'content': message}
            ],
            stream=True,
            extra_body=extra_body
        )
        
        # 处理API响应
        result = format_response(response)
        
        # 返回思考过程和最终答案
        if result['thinking']:
            yield result['thinking']
        yield result['answer']
        
    except Exception as e:
        yield f"Error: {str(e)}"


def add_message(message, chat_history, role='user'):
    """向聊天历史添加新消息"""
    chat_history.append({"role": role, "content": message})
    return chat_history

def clear_history(chat_history):
    """清空聊天历史"""
    return []


def main():
    # 创建Gradio界面
    with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
        gr.Markdown("# 🤖 AI Chat Template")
        gr.Markdown("基于 Gradio 和 ModelScope API 的现代聊天界面")
        
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    bubble_full_width=False,
                    height=300,
                    render=False,
                    show_label=False
                )
            
            with gr.Column(scale=1):
                with gr.Row():
                    clear_btn = gr.Button("🧹 清空")
                
                with gr.Row():
                    api_key_input = gr.Textbox(
                        value=API_KEY if API_KEY else 'your_api_key_here',
                        label="ModelScope API Key",
                        type="password",
                        visible=True
                    )
        
        # 聊天接口
        chat_interface = gr.ChatInterface(
            fn=partial(chatbot_response),
            additional_inputs=[api_key_input],
            chatbot=chatbot,
            additional_inputs_accordion="⚙️ 参数设置",
            examples=[
                ["介绍一下量子计算", ""],
                ["用Python写一个快速排序算法", ""],
                ["分析当前全球气候变化趋势", ""]
            ],
            cache_examples=False
        )
        
        # 绑定清空按钮事件
        clear_btn.click(
            fn=clear_history,
            inputs=[chat_interface.chatbot],
            outputs=[chat_interface.chatbot]
        )
        
        # 添加自定义CSS样式
        gr.HTML(
            """
            <style>
            .chatbot {
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .user {
              background-color: #DCF8C6;
              align-self: flex-end;
              border-radius: 8px 4px 8px 8px;
            }
            .bot {
              background-color: #ECECEC;
              align-self: flex-start;
              border-radius: 4px 8px 8px 8px;
            }
            .chat-message {
              padding: 0.5rem 1rem;
              margin: 0.5rem 0;
              font-size: 1.1rem;
            }
            #chat-container {
              max-width: 800px;
              margin: 0 auto;
            }
            @media (max-width: 600px) {
              .chat-message {
                font-size: 1rem;
              }
            }
            </style>
            """
        )
    
    # 启动应用
    demo.launch(
        share=True  # 生成可分享的公共链接
    )

if __name__ == "__main__":
    main()