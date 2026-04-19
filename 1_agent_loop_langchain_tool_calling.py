from pyexpat.errors import messages

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage , ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


#----- Tools (Langchain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99 ,"headphones":149.95, "keyboard":89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price:float, discount_tier: str)-> float:
    """Apply a Discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount_price(price={price}, discount_tier='{discount_tier}')")
    discount_percentage = {"bronze": 5, "silver": 12,"gold": 23}
    discount = discount_percentage.get(discount_tier,0)
    return round(price *(1-discount/100),2)

#---Agent loop---

@traceable(name="langcahin Agent loop")
def run_agent(question:str):
    tools =[get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature =0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" *60)

    messages = [
        SystemMessage(
            content=(
                "you are a helpful shopping assistant"
                "you have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these instructions exactly:\n"
                "1.Never guess or assume any product price."
                "you must call get_product_price first to get the real price\n"
                "2.If user does not specify the exact product ask which product"
                "they want.\n"
                "3. Only call apply_discount AFTER you have received"
                "a price from the get_product_price. pass the exact price"
                "returned by get_product_price - do not pass a made-up number"
                "4.Never Calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "5. if user does not specify a discount tier, "
                "ask them which tier to use - do not assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- ITERATION {iteration} ----")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # if no tool calls , this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool selected]  {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain agent(.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")
