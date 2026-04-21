import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import vector_stores

load_dotenv()

print("Initializing components....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retreiver = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    
    Question:{question}
    
    Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retreived documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# IMPLEMENTATION 1: Without LCEL (Simple Function-based Approach)

def retreival_chain_without_lcel(query: str):
    """
    Simple retreival chain without LCEL.
    Manually retreives documents, formats them and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    #Step-1: Retreive relevant documents
    docs = retreiver.invoke(query)

    # Step-2: Retreive relevant documents
    context = format_docs(docs)

    # Step-3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question= query)

    #Step-4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    #Step-5: Return the context
    return response.content


# IMPLEMENTATION 2: With LCEL (Langchain Expression language)

def create_retreival_chain_lcell():
    """
    create a retreival chain using LCEL(Langchain Expression language)
    Return a chain that can be invoked with {"question":"..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming :chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with langchain's type system
    - Less code:More concise and readable
    - Reusable: chain can be saved , shared and composed with other chains
    - Better debugging: Langchain provides better observability tools
    """

    retrieval_chain = (
            RunnablePassthrough.assign(
                context = itemgetter("question") | retreiver | format_docs
            )
            | prompt_template | llm | StrOutputParser()
    )

    return retrieval_chain



if __name__ == "__main__":
    print("Retreiving....")

    #Query
    query = "what is pinecone in machine learning?"

    print("\n" + "=" * 70)

    #===================================
    #Option 0: Raw invocation without RAG
    #====================================


    print("IMPLEMENTATION 0: Raw LLM Invocation (No Rag)")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    #===================================
    #Option 1:invocation with RAG(Use implemenatation without LCEL)
    #====================================


    print("IMPLEMENTATION 1: LLM Invocation with Rag but without LCEL")
    print("="*70)
    result_without_lcel = retreival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    #===================================
    #Option 2:Use implemenatation with LCEL)
    #====================================

    print("=" * 70)
    print("IMPLEMENTATION 2: LLM Invocation with Rag and with LCEL")
    print("=" * 70)
    chain_with_lcel = create_retreival_chain_lcell()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
