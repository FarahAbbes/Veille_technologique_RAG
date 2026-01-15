st.info("""
        **✨ No Ollama Required!**

        This version uses HuggingFace Transformers.
        Models download automatically on first use.

        **Install Requirements:**
        ```bash
        pip install streamlit PyPDF2 pandas
        pip install langchain langchain-community
        pip install sentence-transformers
        pip install faiss-cpu transformers torch
        ```

        **Model Sizes:**
        - Small: ~300MB (fast)
        - Base: ~1GB (recommended)
        - Large: ~3GB (best quality)
        """)
