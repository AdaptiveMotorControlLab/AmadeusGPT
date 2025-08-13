export streamlit_app=True
app:

	uv run --with streamlit streamlit run amadeusgpt/app.py --server.fileWatcherType none --server.maxUploadSize 1000
