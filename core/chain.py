"""History aware RAG chain using LangChain"""

from config import TOP_K
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
)
from langchain_openai import ChatOpenAI

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and latest user question,"
            "rewrite the question as a fully self contained standalone question"
            "that can be understood without chat history."
            "Do NOT answer the question - only reformulate it if needed,"
            "otherwise return as it-is"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for question answering tasks"
            "Use only the follwing retrived context to answer the question."
            "If the answer is not present in the context , say you don't know."
            "Be clear and concise. \n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def _format_docs(docs:list[Document]) -> str:
    """Concatenate document page content into a single context string"""
    return "\n\n".join(d.page_content for d in docs)



def build_rag_chain(vectore_store: Chroma, api_key : str, model: str) -> Runnable:
    """Build and return a history aware conversational RAG chain"""

    llm = ChatOpenAI(api_key=api_key, model = model, temperature=0)
    retriver = vectore_store.as_retriever(search_kwargs={"k": TOP_K})
    contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    def retrive_with_history(input_dict: dict) -> list[Document]:
        """Reformulate the question with history and then retrive docs"""

        question: str = input_dict["input"]
        history : list = input_dict.get("chat_history", [])
        if history:
            question  = contextualize_chain.invoke(
                {"input": question, "chat_history": history}
            )
        return retriver.invoke(question)
    
    retrive_step = RunnablePassthrough.assign(context=RunnableLambda(retrive_with_history))
    
    answer_step = RunnableParallel(
        answer = (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | _QA_PROMPT
            | llm
            | StrOutputParser()
        ),
        context = lambda x: x["context"],
    )

    return retrive_step | answer_step
