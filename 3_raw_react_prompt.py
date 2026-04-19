# CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
import re
import inspect
from pydoc import getdoc

from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


#----- Tools (Langchain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99 ,"headphones":149.95, "keyboard":89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price:float, discount_tier: str)-> float:
    """Apply a Discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount_price(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentage = {"bronze": 5, "silver": 12,"gold": 23}
    discount = discount_percentage.get(discount_tier,0)
    return round(price *(1-discount/100),2)

tools = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature}- {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ",".join(tools.keys())


react_prompt = f"""
STRICT RULES - you must follow these instructions exactly:
                1.Never guess or assume any product price.
                you must call get_product_price first to get the real price.
                2.If user does not specify the exact product ask which product.
                they want.
                3. Only call apply_discount AFTER you have received
                a price from the get_product_price. pass the exact price.
                returned by get_product_price - do not pass a made-up number.
                4.Never Calculate discounts yourself using math.
                Always use the apply_discount tool.
                5. if user does not specify a discount tier,
                ask them which tier to use - do not assume one.
                
                Answer the following questions as best you can. You have access to the following tools:
                
                {tool_descriptions}
                
                Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: tha action to take, should be one of [{tool_names}]
                Action Input: the input to the action, as comma separated values
                Observation: the result of the action
                ...(This Thought/Action/Action input/Observation can repeat N times)
                Thought: I now know the final answer
                Final answer: the final answer to the original input question
                
                Begin!
                
                Question:{{question}}
                Thought:"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)

#---Agent loop---

@traceable(name="langcahin Agent loop")
def run_agent(question:str):

    print(f"Question: {question}")
    print("="*60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- ITERATION {iteration} ----")
        full_prompt = prompt + scratchpad

        # stop token prevents The LLM from generating its own observation -
        # we inject the real tool result instead.

        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role":"user","content":full_prompt}],
            options= {"stop":["\nObservation"], "temperature": 0}
        )

        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # Difference 7: History is one growing string re-sent every iteration( replaces messages.append)

        scratchpad += f"{output}\n Observation: {observation}\n Thought:"

    print("ERROR: Max iterations reached without a final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain agent(.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")
