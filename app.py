import os
import gradio as gr
from openai import OpenAI
import time

# 初始化OpenAI客户端
client = OpenAI(
    api_key="填写自己的密钥哦~",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3"
)

# 游戏状态
class GameState:
    def __init__(self):
        self.questions_asked = 0
        self.max_questions = 20
        self.game_history = []
        self.target_object = None
        self.is_game_over = False
        self.current_question = None
        self.is_game_started = False
        self.ai_score = 0
        self.human_score = 0

game_state = GameState()

def stream_response(response, history_text):
    """流式输出响应"""
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response, history_text
    return full_response, history_text

def set_target_object(target):
    """设置目标物体并开始游戏"""
    if not target.strip():
        return "请输入目标物体！", "", "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""
    
    game_state.target_object = target.strip()
    game_state.is_game_started = True
    game_state.questions_asked = 0
    game_state.game_history = []
    game_state.is_game_over = False
    game_state.current_question = None
    
    # 让AI提出第一个问题
    prompt = """你正在玩20问游戏。请提出第一个问题来猜测玩家心中的物体。
问题应该是一个简单的"是/否"问题，比如"它是活物吗？"、"它比汽车大吗？"等。
请先思考一下，然后只输出问题，不要输出其他内容。"""
    
    try:
        yield "游戏开始！", "\n".join(game_state.game_history), "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", "AI正在思考第一个问题..."
        
        response = client.chat.completions.create(
            model="ernie-x1-turbo-32k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
            stream=True
        )
        
        first_question = ""
        thinking_process = "AI正在思考第一个问题...\n\n"
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                thinking_process += chunk.choices[0].delta.reasoning_content
                yield "游戏开始！", "\n".join(game_state.game_history), "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process
            if chunk.choices[0].delta.content:
                first_question += chunk.choices[0].delta.content
        
        game_state.current_question = first_question.strip()
        game_state.game_history.append(f"AI问题 {game_state.questions_asked + 1}: {first_question.strip()}")
        yield f"游戏开始！\nAI的第一个问题：{first_question.strip()}", "\n".join(game_state.game_history), "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process + "\n\n思考完成！"
    except Exception as e:
        yield f"发生错误: {str(e)}", "", "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""

def answer_question(answer):
    """处理玩家的回答"""
    if not game_state.is_game_started:
        return "请先设置目标物体！", "", "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""
    
    if game_state.is_game_over:
        return "游戏已结束，请开始新游戏！", "", "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""
    
    if not answer.strip():
        return "请输入你的回答！", "", "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""
    
    # 记录玩家的回答
    game_state.game_history.append(f"玩家回答: {answer}")
    game_state.questions_asked += 1
    
    # 如果是最后一轮，让AI进行最终猜测
    if game_state.questions_asked >= game_state.max_questions:
        game_state.is_game_over = True
        
        # 让AI进行最终猜测
        guess_prompt = f"""基于之前的对话：
{chr(10).join(game_state.game_history)}

这是最后一轮了，请根据所有信息，给出你的最终猜测。
请先分析一下已有的信息，然后给出你的猜测。
只输出一个具体的物体名称，不要输出其他内容。"""
        
        try:
            yield "游戏即将结束...", "\n".join(game_state.game_history), "", "20/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", "AI正在分析所有信息并做出最终猜测..."
            
            guess_response = client.chat.completions.create(
                model="ernie-x1-turbo-32k",
                messages=[{"role": "user", "content": guess_prompt}],
                temperature=0.7,
                max_tokens=50,
                stream=True
            )
            
            guess = ""
            thinking_process = "AI正在分析所有信息并做出最终猜测...\n\n"
            for chunk in guess_response:
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    thinking_process += chunk.choices[0].delta.reasoning_content
                    yield "游戏即将结束...", "\n".join(game_state.game_history), "", "20/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process
                if chunk.choices[0].delta.content:
                    guess += chunk.choices[0].delta.content
            
            guess = guess.strip()
            game_state.game_history.append(f"AI最终猜测: {guess}")
            
            # 判断AI的猜测是否正确
            if guess.lower().strip() == game_state.target_object.lower().strip():
                game_state.ai_score += 1
                result = f"AI猜对了！目标物体就是：{game_state.target_object}\nAI得1分！"
            else:
                game_state.human_score += 1
                result = f"AI猜错了！目标物体是：{game_state.target_object}\n人类得1分！"
            
            history_text = "\n".join(game_state.game_history)
            yield result, history_text, "", "20/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process + "\n\n思考完成！"
            return
        except Exception as e:
            yield f"发生错误: {str(e)}", "", "", "20/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""
            return
    
    # 如果不是最后一轮，让AI继续提问
    next_prompt = f"""基于之前的对话：
{chr(10).join(game_state.game_history)}

请分析这些问答，提出下一个问题来猜测玩家心中的物体。
问题应该是一个简单的"是/否"问题，要基于之前的回答来缩小范围，但也要注意，有时玩家并不知道这个物体应该选是还是否，可能会出现回答错误，因此要考虑这种情况。
请先思考一下，然后只输出问题，不要输出其他内容。"""
    
    try:
        yield "AI正在思考...", "\n".join(game_state.game_history), "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", f"AI正在分析第{game_state.questions_asked}轮的回答并思考下一个问题..."
        
        next_response = client.chat.completions.create(
            model="ernie-x1-turbo-32k",
            messages=[{"role": "user", "content": next_prompt}],
            temperature=0.7,
            max_tokens=100,
            stream=True
        )
        
        next_question = ""
        thinking_process = f"AI正在分析第{game_state.questions_asked}轮的回答并思考下一个问题...\n\n"
        for chunk in next_response:
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                thinking_process += chunk.choices[0].delta.reasoning_content
                yield "AI正在思考...", "\n".join(game_state.game_history), "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process
            if chunk.choices[0].delta.content:
                next_question += chunk.choices[0].delta.content
        
        next_question = next_question.strip()
        game_state.current_question = next_question
        game_state.game_history.append(f"AI问题 {game_state.questions_asked + 1}: {next_question}")
        
        history_text = "\n".join(game_state.game_history)
        yield f"AI的问题 {game_state.questions_asked + 1}/20: {next_question}", history_text, "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", thinking_process + "\n\n思考完成！"
    except Exception as e:
        yield f"发生错误: {str(e)}", "", "", f"{game_state.questions_asked}/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", ""

def reset_game():
    """重置游戏"""
    game_state.questions_asked = 0
    game_state.game_history = []
    game_state.is_game_over = False
    game_state.target_object = None
    game_state.current_question = None
    game_state.is_game_started = False
    return "游戏已重置！请设置新的目标物体。", "", "", "0/20", f"AI: {game_state.ai_score} - 人类: {game_state.human_score}", "游戏已重置，等待开始新游戏..."

# 自定义CSS样式
custom_css = """
.round-counter {
    font-size: 24px !important;
    font-weight: bold !important;
    color: #4a90e2 !important;
    padding: 10px 20px !important;
    background-color: #f5f5f5 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    margin-left: auto !important;
}

.score-counter {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #2c3e50 !important;
    padding: 8px 16px !important;
    background-color: #ecf0f1 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    margin-left: 20px !important;
}

.thinking {
    color: #666 !important;
    font-style: italic !important;
}

.ai-thinking {
    font-family: 'Courier New', monospace !important;
    background-color: #f8f9fa !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border-left: 4px solid #4a90e2 !important;
    margin: 10px 0 !important;
    white-space: pre-wrap !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}
"""

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        gr.Markdown("# 🎮 知你所想")
        round_counter = gr.Markdown("0/20", elem_classes=["round-counter"])
        score_counter = gr.Markdown("AI: 0 - 人类: 0", elem_classes=["score-counter"])
    gr.Markdown("### 20个问题，猜中你所想的东西，来挑战一下吧！")
    
    with gr.Row():
        with gr.Column(scale=1):
            target_input = gr.Textbox(
                label="输入你想让AI猜的物体",
                placeholder="例如：长颈鹿",
                lines=2
            )
            set_target_button = gr.Button("设置目标物体", variant="primary")
        with gr.Column(scale=1):
            answer_input = gr.Textbox(
                label="回答AI的问题",
                placeholder="输入：是、否、或不知道",
                lines=2
            )
            answer_button = gr.Button("回答", variant="primary")
    

    with gr.Row():
        reset_button = gr.Button("开始新游戏", variant="stop")
    
    with gr.Row():
        ai_thinking = gr.Textbox(
            label="AI思考过程",
            lines=5,
            elem_classes=["ai-thinking"],
            interactive=False
        )
    
    with gr.Row():
        output = gr.Textbox(label="游戏状态", lines=10)
        history = gr.Textbox(label="游戏历史", lines=10)
        error = gr.Textbox(label="错误信息", lines=10)
    
    # 设置事件处理
    set_target_button.click(
        set_target_object,
        inputs=[target_input],
        outputs=[output, history, error, round_counter, score_counter, ai_thinking],
        queue=True
    )
    
    answer_button.click(
        answer_question,
        inputs=[answer_input],
        outputs=[output, history, error, round_counter, score_counter, ai_thinking],
        queue=True
    )
    
    reset_button.click(
        reset_game,
        outputs=[output, history, error, round_counter, score_counter, ai_thinking]
    )

if __name__ == "__main__":
    demo.queue().launch() 