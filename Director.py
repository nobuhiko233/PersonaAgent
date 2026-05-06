from langchain_deepseek import ChatDeepSeek
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))


# 定义状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str
    user_profile: dict


# 构建图
builder = StateGraph(State)


# 定义路由结构
class RouteSchema(BaseModel):
    """用户查询的意图路由方案"""

    type: Literal["movie", "game", "anime", "other"] = Field(
        description="意图分类结果，必须且只能是 'movie', 'game', 'anime', 'other' 这四个值中的一个。"
    )


# 定义节点
async def supervisor_node(state: State):
    system_prompt = """
    <role>
    你是一个高精度的意图识别引擎（Intent Routing Supervisor），负责将用户的自然语言查询分发给最匹配的下游 Agent。
    </role>

    <skills>
    - 意图提纯：剥离用户情绪，精准锁定查询的核心诉求。
    - 边界判定：严格评估查询是否属于预设的三个垂直领域（电影、游戏、动漫）。
    </skills>

    <rules>
    1. 电影意图：涉及院线、影视演员、导演、剧情解析等 -> 返回 movie
    2. 游戏意图：涉及电子游戏、Steam数据、玩法推荐、游戏配置等 -> 返回 game
    3. 动漫意图：涉及二次元文化、番剧、漫画、轻小说、ACG配音等 -> 返回 anime
    4. 兜底策略：只要不属于上述三类的任何其他闲聊或提问 -> 必须返回 other
    </rules>

    <output_format>
    你只能输出 [movie, game, anime, other] 中的一个英文单词，绝对不能包含任何标点符号、解释或分析过程。
    </output_format>
    """
    recent_messages = state["messages"][-5:]
    prompts = [SystemMessage(content=system_prompt)] + recent_messages
    structured_llm = llm.with_structured_output(RouteSchema)
    res = await structured_llm.ainvoke(prompts)
    return {"type": res.type}


async def anime_node(state: State):
    profile = state.get("user_profile", {})
    profile_str = f"已知用户偏好: {profile}" if profile else "目前暂无用户偏好数据。"
    system_prompt = """
    <role>
    你是具有百科级别 ACG（动画、漫画、游戏）知识的二次元数据分析师。
    </role>

    <context>
    {profile_str} 
    请在推荐动漫时，优先考虑上述已知的用户偏好。
    </context>

    <skills>
    - Bangumi 数据调用：熟练使用 `bangumi_tv` 技能查询高精度的影视、番剧库。
    - 元数据提炼：从庞杂的工具返回数据中，提取出用户真正关心的信息（如：声优、放送日期、评分）。
    </skills>

    <constraints>
    1. 遇到具体番剧或角色的细节提问，务必先调用技能查询，不要依赖你的底层训练权重。
    2. 用户体验优先：过滤掉工具返回结果中的 JSON 格式或技术代码，转化为自然语言。
    3. 内容精炼：回答总字数严格控制在 200 字以内，直击痛点。
    </constraints>
    """
    recent_messages = state["messages"][-5:]
    prompts = [SystemMessage(content=system_prompt)] + recent_messages
    client = MultiServerMCPClient(
        {
            "bangumi-tv": {
                "command": os.getenv("MCP_UV_PATH"),
                "transport": "stdio",
                "args": [
                    "--directory",
                    os.getenv("MCP_BANGUMI_PATH"),
                    "run",
                    "main.py",
                ],
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent(model=llm, tools=tools)
    res = await agent.ainvoke({"messages": prompts})
    return {
        "messages": [AIMessage(content=res["messages"][-1].content)],
        "type": "anime",
    }


async def game_node(state: State):
    profile = state.get("user_profile", {})
    profile_str = f"已知用户偏好: {profile}" if profile else "目前暂无用户偏好数据。"
    system_prompt = """
    <role>
    你是深谙 Steam 生态的高级游戏策展人（Steam Curator Agent）。
    </role>

    <context>
    {profile_str} 
    请在推荐游戏时，优先考虑上述已知的用户偏好。
    </context>

    <skills>
    - 检索增强生成 (RAG)：通过 `steam_game_search` 技能，精准提取 2025 年 Steam 本地数据库中的信息。
    - 数据对齐：将用户的模糊描述与数据库中的游戏特征（标签、价格、好评率）进行匹配。
    </skills>

    <constraints>
    1. 必须优先调用 `steam_game_search` 技能获取数据。
    2. 事实隔离：你的回答必须【严格基于检索到的数据】，绝不能凭空捏造（Hallucinate）游戏存在或篡改游戏数据。
    3. 如果检索结果为空或不相关，请诚实地回答：“在目前的 2025 Steam 游戏库中未找到相关推荐”，不要强行编造。
    </constraints>

    <style>
    像专业的游戏评测编辑一样说话。推荐游戏时，请列出游戏名，并用一两句简练的话说明推荐理由。
    </style>
    """
    recent_messages = state["messages"][-5:]
    prompts = [SystemMessage(content=system_prompt)] + recent_messages
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_PATH"),
        model_kwargs={"local_files_only": True},
    )
    vector_store = Chroma(
        persist_directory="./steam_chroma_db", embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    game_tool = create_retriever_tool(
        retriever,
        "steam_game_search",
        "搜索Steam游戏数据库。当你需要了解游戏信息、推荐游戏时使用此工具。",
    )
    tools = [game_tool]
    agent = create_agent(model=llm, tools=tools)
    res = await agent.ainvoke({"messages": prompts})
    return {
        "messages": [AIMessage(content=res["messages"][-1].content)],
        "type": "game",
    }


async def movie_node(state: State):
    profile = state.get("user_profile", {})
    profile_str = f"已知用户偏好: {profile}" if profile else "目前暂无用户偏好数据。"
    system_prompt = """
    <role>
    你是资深的影视鉴赏家与数据向导。
    </role>

    <context>
    {profile_str} 
    请在推荐电影时，优先考虑上述已知的用户偏好。
    </context>

    <skills>
    - 影视流派分类与风格解析。
    - 观影偏好推断。
    </skills>

    <constraints>
    1. 提供客观的影片信息（上映年份、导演、主演）以及主观的风格评价。
    2. 如果用户寻求资源下载链接，请委婉拒绝并引导至正版流媒体平台。
    </constraints>
    """
    recent_messages = state["messages"][-5:]
    prompts = [SystemMessage(content=system_prompt)] + recent_messages
    res = await llm.ainvoke(prompts)
    return {
        "messages": [AIMessage(content=res.content)],
        "type": "movie",
    }


async def other_node(state: State):
    return {
        "messages": [HumanMessage(content="我暂时无法回答这个问题。")],
        "type": "other",
    }


# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("movie_node", movie_node)
builder.add_node("game_node", game_node)
builder.add_node("anime_node", anime_node)
builder.add_node("other_node", other_node)


# 定义路由函数
def routing_func(state: State):
    if state["type"] == "movie":
        return "movie_node"
    elif state["type"] == "game":
        return "game_node"
    elif state["type"] == "anime":
        return "anime_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"


# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges(
    "supervisor_node",
    routing_func,
    ["movie_node", "game_node", "anime_node", "other_node", END],
)
builder.add_edge("movie_node", END)
builder.add_edge("game_node", END)
builder.add_edge("anime_node", END)
builder.add_edge("other_node", END)
