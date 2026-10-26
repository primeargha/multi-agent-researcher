# Wire agents together and pass data through shared state.
from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_search_pipeline(topic: str) -> dict:
    """Run full research flow: search → reader → writer → critic."""

    # Shared bag of results for this run (search → reader → writer → critic).
    state = {}

    # --- Step 1: find good sources (no scraping yet) ---
    print("\n" + " =" * 50)
    print("Step 1: Search agent is working ...")
    print("=" * 50)

    # Create the search agent (LLM + web_search tool).
    search_agent = build_search_agent()

    # Ask the agent to search. It may call tools, then give a final answer.
    # "user" = the task we give it this turn.
    search_response = search_agent.invoke({
        "messages": [
            ("user", f"Find recent, reliable and detailed information about: {topic}")
        ]
    })

    # Keep only the last message (final answer) for the next step.
    state["search_results"] = search_response["messages"][-1].content

    print("\n search results ", state["search_results"])

    # reader agent working
    print("\n" + " =" * 50)
    print("Step 2: Reader agent is working ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_response = reader_agent.invoke({
        "messages": [
            ("user", 
            f"Based on the following search result about '{topic}',"
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
            )
        ]
    })

    state["scraped_content"] = reader_response["messages"][-1].content

    print("\n scraped content ", state["scraped_content"])

    # writer agent working
    print("\n" + " =" * 50)
    print("Step 3: Writer agent is working ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    writer_response = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = writer_response

    print("\n report ", state["report"])

    # critic agent working
    print("\n" + " =" * 50)
    print("Step 4: Critic agent is working ...")
    print("=" * 50)

    # Critic prompt expects: topic, research, draft (not "report")
    critic_response = critic_chain.invoke({
        "topic": topic,
        "research": research_combined,
        "draft": state["report"],
    })

    state["feedback"] = critic_response

    print("\n feedback ", state["feedback"])

    return state
    
