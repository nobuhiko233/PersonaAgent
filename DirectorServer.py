import asyncio
from langchain_core.messages import HumanMessage
from Director import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def main():
    config = {"configurable": {"thread_id": "1"}}
    query1 = "我想喜欢搞笑轻松类的电影，给我来点推荐。"
    print(f"User: {query1}")
    async with AsyncSqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        res1 = await graph.ainvoke(
            {"messages": [HumanMessage(content=query1)]}, config=config
        )
        print(f"Agent: {res1['messages'][-1].content}\n")
        query2 = "给我推荐些动漫。"
        print(f"User: {query2}")
        res2 = await graph.ainvoke(
            {"messages": [HumanMessage(content=query2)]}, config=config
        )
        print(f"Agent: {res2['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
