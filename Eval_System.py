import asyncio
import os
from langsmith import Client
from langsmith.evaluation import aevaluate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from Director import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "PersonaAgent"

client = Client()

# ==========================================
# 1. 定义测试数据集 
# ==========================================
dataset_name = "PersonaAgent-Routing-QA"

# 如果数据集不存在，则自动创建
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="用于测试意图路由与基础回答质量的黄金数据集"
    )
    # 添加测试用例 (Inputs: 用户提问, Outputs: 期望的分类和参考答案)
    test_cases = [
        ("给我推荐几部搞笑电影", "movie", "推荐《三傻大闹宝莱坞》等喜剧电影"),
        ("Windows平台有什么好玩的动作游戏", "game", "推荐Steam上的动作游戏"),
        ("进击的巨人是谁画的", "anime", "谏山创"),
        ("今天天气真好", "other", "我暂时无法回答这个问题"),
    ]
    for q, expected_route, expected_ans in test_cases:
        client.create_example(
            inputs={"question": q},
            outputs={"expected_route": expected_route, "expected_answer": expected_ans},
            dataset_id=dataset.id,
        )
    print(f"✅ 数据集 {dataset_name} 创建成功！")

# ==========================================
# 2. 定义目标函数 (将我们的 Graph 包装成标准形式)
# ==========================================
async def predict_agent_response(inputs: dict):
    """接收 LangSmith 的输入，喂给我们的 LangGraph，并返回结果"""
    question = inputs["question"]
    # 为了评测，我们每次使用一个随机的 thread_id 保证独立性
    config = {"configurable": {"thread_id": f"eval_{os.urandom(4).hex()}"}}
    
    async with AsyncSqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        # 记录运行前的路由状态
        final_state = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config=config)
        
        return {
            "actual_answer": final_state["messages"][-1].content,
            "actual_route": final_state["type"] 
        }

# ==========================================
# 3. 定义自定义评估器 (Metrics)
# ==========================================
def exact_route_match(run, example) -> dict:
    """硬性指标：测试 Supervisor 的路由是否 100% 正确"""
    expected_route = example.outputs["expected_route"]
    actual_route = run.outputs["actual_route"]
    
    score = 1.0 if expected_route == actual_route else 0.0
    return {"key": "route_accuracy", "score": score}

def llm_judge_helpfulness(run, example) -> dict:
    """柔性指标：使用大模型作为裁判 (LLM-as-a-Judge) 评价回答质量"""
    expected_ans = example.outputs["expected_answer"]
    actual_ans = run.outputs["actual_answer"]
    
    # 用一个轻量级大模型来打分
    judge_llm = ChatDeepSeek(model="deepseek-chat", api_key="sk-38ddaf0c7cc14fabaa15d657f32ebb38")
    
    prompt = f"""
    你是一个严厉的评分考官。请根据[期望答案]，评估[实际回答]的质量。
    [期望答案]: {expected_ans}
    [实际回答]: {actual_ans}
    
    如果实际回答准确、有帮助且没有幻觉，请回复"1"；否则回复"0"。只能回复数字，不要解释。
    """
    res = judge_llm.invoke(prompt)
    
    # 提取分数
    try:
        score = float(res.content.strip())
    except:
        score = 0.0
        
    return {"key": "helpfulness_score", "score": score}

# ==========================================
# 4. 执行自动化批量评测
# ==========================================
async def run_eval():
    print("🚀 开始执行 LangSmith 自动化评测...")
    experiment_results = await aevaluate(
        predict_agent_response,
        data=dataset_name,
        evaluators=[exact_route_match, llm_judge_helpfulness],
        experiment_prefix="DeepSeek-Agent-Eval-",
        metadata={"version": "1.0", "prompt_technique": "Skill-Based Prompting"}
    )
    print("✅ 评测完成！请前往 LangSmith 网页端查看详细对比报告。")

if __name__ == "__main__":
    asyncio.run(run_eval())