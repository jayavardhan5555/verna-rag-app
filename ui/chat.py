import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

def render_chat() -> None:
    st.title("VERNA RAG CHAT")
    st.caption("upload a pdf/word document or provide a web URL , then ask questions about it")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Source chunks", expanded=False):
                    for i , chunk in enumerate(msg["sources"],1):
                        st.markdown(f"**[{i}]** {chunk[:500]}")

    user_input: str | None = st.chat_input("Ask a question about your doc or url", disabled="rag_chain" not in st.session_state)

    if not user_input:
        return
    
    if "rag_chain" not in st.session_state:
        st.warning("process a doc or url")
        return
    
    with st.chat_message("assistant"):
        with st.spinner("Verna Thinking ...."):
            try:
                result = st.session_state.rag_chain.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    }
                )

                answer : str = result["answer"]
                source_docs = result.get("context", [])
                source_texts:list[str] = [doc.page_content for doc in source_docs]

                st.markdown(answer)

                if source_texts:
                    with st.expander("Source Chunks", expanded= False):
                        for i, chunk in enumerate(source_texts,1):
                            st.markdown(f"**[{i}]** {chunk[:500]}")
                        
                st.session_state.chat_history.extend(
                    [HumanMessage(content=user_input), AIMessage(content=answer)]
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_texts
                    }
                )
            
            except Exception as ex:
                error_message = f"Error {ex}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant", "content" : error_message
                    }
                )
                    