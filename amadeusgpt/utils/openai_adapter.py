import openai
from amadeusgpt.utils.embedding_adapter import EmbeddingModel

class OpenAIAdapter:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenAIAdapter, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: str = None):
        if self._initialized:
            return

        if api_key is None:
            raise ValueError("API Key is required")
        
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = None
        self.is_openrouter = False
        self._initialized = True
        self._initialize_client()
        self.client.embeddings = EmbeddingModel(provider_type=self.get_provider_type()).embeddings

    def __call__(self, *args, **kwargs):
        if self.client is None:
            raise RuntimeError("Client is not initialized")
        return self.client

    def _initialize_client(self):
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
            )
            client.models.list()
            self.client = client
        except openai.AuthenticationError:
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                client.models.list()
                self.client = client
                self.is_openrouter = True
            except openai.AuthenticationError:
                raise openai.AuthenticationError("Invalid API key")

    def validate(self):
        return self.client is not None

    def get_provider_type(self):
        return "openrouter" if self.is_openrouter else "openai"

    def get_client(self):
        return self.client
        
    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False
        
