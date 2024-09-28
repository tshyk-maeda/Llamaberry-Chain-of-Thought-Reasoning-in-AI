import os
import asyncio
import gradio as gr
from groq import AsyncGroq
import time

# 必要なライブラリをインポートしています。
# os: 環境変数を取得するために使用
# asyncio: 非同期プログラミングをサポート
# gradio: ウェブインターフェースを作成するために使用
# AsyncGroq: Groq APIの非同期クライアント
# time: 処理時間を計測するために使用

# Groqクライアントの初期化
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
# 環境変数からGroq APIキーを取得し、非同期クライアントを作成します。
# セキュリティのため、APIキーは環境変数に保存することが推奨されます。

# 使用するモデルを定義
model = "llama-3.1-70b-versatile"
# Groq APIで使用する言語モデルを指定します。

# 初期システムプロンプト、フォローアップシステムプロンプト、統合プロンプトの定義
initial_system_prompt = """You are an AI assistant capable of detailed, step-by-step thinking. When presented with a question or problem, break down your thought process into clear, logical steps. For each step, explain your reasoning. Conclude with a final answer. Use the following markdown structure:

## Reasoning
1. [First step]
   **Explanation:** [Detailed explanation of this step]
2. [Second step]
   **Explanation:** [Detailed explanation of this step]
...

## Answer
[Final answer]

Be comprehensive and show your reasoning clearly."""
# このプロンプトは、AIアシスタントに対する指示を含んでいます。
# このプロンプトは、質問や問題に対して段階的な思考プロセスを実行するように指示し、各ステップで理由を説明し、最終的な答えを提供するように指示しています。

followup_system_prompt = """You are an AI assistant tasked with analyzing and improving upon previous problem-solving steps. Review the original query and the previous turns of reasoning, then provide a new perspective or deeper analysis. Use the following markdown structure:

## Critique
[Provide a brief critique of the previous reasoning, highlighting its strengths and potential weaknesses]

## New Reasoning
1. [First step of new or refined approach]
   **Explanation:** [Detailed explanation of this step, referencing the previous reasoning if relevant]
2. [Second step of new or refined approach]
   **Explanation:** [Explanation of how this step builds upon or differs from the previous thinking]
...

## Updated Answer
[Updated answer based on this new analysis]

Be critical yet constructive, and strive to provide new insights or improvements."""
# このプロンプトは、AIアシスタントに対する指示を含んでいます。
# このプロンプトは、前回の推論を分析し、改善するように指示し、新しい視点や深い分析を提供するように指示しています。

synthesis_prompt = """You are an AI assistant tasked with synthesizing multiple turns of reasoning into a final, comprehensive answer. You will be presented with three different turns of reasoning for solving a problem. Your task is to:

1. Analyze each turn, considering its strengths and weaknesses.
2. Compare and contrast the different methods.
3. Synthesize the insights from all turns into a final, well-reasoned answer.
4. Provide a concise, clear final answer that a general audience can understand.

Use the following markdown structure:

## Analysis of Turns
[Provide a brief analysis of each turn of reasoning]

## Comparison
[Compare and contrast the turns, highlighting key differences and similarities]

## Final Reasoning
[Provide a final, synthesized reasoning process that combines the best insights from all turns]

## Comprehensive Final Answer
[Comprehensive final answer]

## Concise Answer
[A brief, clear, and easily understandable version of the final answer, suitable for a general audience. This should be no more than 2-3 sentences.]

Be thorough in your analysis and clear in your reasoning process."""
# このプロンプトは、AIアシスタントに対する指示を含んでいます。
# このプロンプトは、複数のターンの推論を統合して最終的な包括的な答えを生成するように指示しています。
# このプロンプトは、各ターンを分析し、その強みと弱みを考慮し、異なる方法を比較・対照させ、すべてのターンからの洞察を統合して最終的な包括的な答えを生成するように指示しています。
# また、一般の聴衆が理解できる簡潔で明確な最終答えを提供するように指示しています。

async def call_llm(messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:
    """Groq APIを呼び出す非同期関数"""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
# この関数は、Groq APIを非同期で呼び出し、AIモデルからの応答を取得します。
# temperature: 応答のランダム性を制御（高いほどランダム）
# max_tokens: 生成するテキストの最大トークン数

async def generate_turn(query: str, previous_turns: list = None) -> str:
    """単一のターンの推論を生成する非同期関数"""
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    if is_first_turn:
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        previous_content = "\n\n".join(previous_turns)
        messages = [{
            "role": "system",
            "content": followup_system_prompt
        }, {
            "role": "user",
            "content": f"Original Query: {query}\n\nPrevious Turns:\n{previous_content}\n\nProvide the next turn of reasoning."
        }]
    return await call_llm(messages)
# この関数は、単一のターンの推論を生成します。
# 最初のターンか後続のターンかによって、異なるプロンプトを使用します。

async def synthesize_turns(query: str, turns: list) -> str:
    """複数のターンの推論を統合する非同期関数"""
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role": "user",
        "content": f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    return await call_llm(messages)
# この関数は、複数のターンの推論を統合して最終的な回答を生成します。

async def full_cot_reasoning(query: str) -> tuple:
    """完全なChain of Thought推論を実行する非同期関数"""
    start_time = time.time()
    turns = []
    turn_times = []
    full_output = f"# Chain of Thought Reasoning\n\n## Original Query\n{query}\n\n"

    # 3ターンの推論を生成
    for i in range(3):
        turn_start = time.time()
        turn = await generate_turn(query, turns)
        turns.append(turn)
        turn_times.append(time.time() - turn_start)
        full_output += f"## Turn {i+1}\n{turn}\n\n"

    # 統合ステップ
    mid_time = time.time()
    synthesis = await synthesize_turns(query, turns)
    full_output += f"## Synthesis\n{synthesis}\n\n"
    end_time = time.time()

    # タイミング情報の計算と出力への追加
    timing = {
        'turn_times': turn_times,
        'total_turns_time': mid_time - start_time,
        'synthesis_time': end_time - mid_time,
        'total_time': end_time - start_time
    }
    full_output += f"## Timing Information\n"
    full_output += f"- Turn 1 Time: {timing['turn_times'][0]:.2f}s\n"
    full_output += f"- Turn 2 Time: {timing['turn_times'][1]:.2f}s\n"
    full_output += f"- Turn 3 Time: {timing['turn_times'][2]:.2f}s\n"
    full_output += f"- Total Turns Time: {timing['total_turns_time']:.2f}s\n"
    full_output += f"- Synthesis Time: {timing['synthesis_time']:.2f}s\n"
    full_output += f"- Total Time: {timing['total_time']:.2f}s\n"

    return full_output
# この関数は、完全なChain of Thought推論プロセスを実行します。
# 3つのターンの推論を生成し、それらを統合し、タイミング情報を追加します。

def gradio_interface(query: str) -> str:
    """Gradioインターフェース関数"""
    return asyncio.run(full_cot_reasoning(query))
# この関数は、Gradioインターフェースから呼び出され、非同期関数を同期的に実行します。

# Gradioインターフェースの作成
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="Enter your question or problem")],
    outputs=[gr.Markdown(label="Chain of Thought Reasoning")],
    title="Multi-Turn Chain of Thought Reasoning with Final Synthesis",
    description=
    "Enter a question or problem to see multiple turns of reasoning, followed by a final synthesized answer."
)

# インターフェースの起動
iface.launch(share=True)
# Gradioインターフェースを作成し、起動します。
# share=Trueにすることで、一時的な公開URLが生成されます。