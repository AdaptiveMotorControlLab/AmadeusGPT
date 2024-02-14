export streamlit_app=True
app:

	streamlit run amadeusgpt/app.py --server.fileWatcherType none --server.maxUploadSize 1000
