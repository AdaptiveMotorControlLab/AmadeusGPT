import os


def get_api_key(streamlit_session_state):
    if streamlit_session_state.get("OPENAI_API_KEY"):
        return streamlit_session_state.get("OPENAI_API_KEY")
    elif streamlit_session_state.get("OPENROUTER_API_KEY"):
        return streamlit_session_state.get("OPENROUTER_API_KEY")
    elif streamlit_session_state.get("openAI_token"):
        return streamlit_session_state.get("openAI_token")
    elif os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_API_KEY")
    elif os.environ.get("OPENROUTER_API_KEY"):
        return os.environ.get("OPENROUTER_API_KEY")
    else:
        raise ValueError("No API key found")
    