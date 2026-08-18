"""Single Streamlit entrypoint for every configured runtime profile."""

from distributed_rag.interfaces.streamlit_app import run_streamlit_app


if __name__ == "__main__":
    run_streamlit_app()
