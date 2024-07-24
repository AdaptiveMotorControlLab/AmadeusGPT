import os
import pickle

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from amadeusgpt.programs.api_registry import INTEGRATION_API_REGISTRY

client = OpenAI()


class IntegrationModuleHub:
    def __init__(self):
        self.amadeus_root = os.path.dirname(os.path.realpath(__file__))

    def save_embeddings(self):
        result = {}
        model = "text-embedding-3-small"
        for module_name, module_info in INTEGRATION_API_REGISTRY.items():
            docstring = module_info["description"]
            text = docstring.replace("\n", " ")
            embedding = (
                client.embeddings.create(input=[text], model=model).data[0].embedding
            )
            result[module_name] = embedding
        if len(result) > 0:
            with open(
                os.path.join(self.amadeus_root, "modules_embedding.pickle"), "wb"
            ) as file:
                pickle.dump(result, file)
        else:
            print("No module found")

    def match_module(self, query):
        model = "text-embedding-3-small"

        query_embedding = (
            client.embeddings.create(input=[query], model=model).data[0].embedding
        )

        if not os.path.exists(
            os.path.join(self.amadeus_root, "modules_embedding.pickle")
        ):
            self.save_embeddings()

        with open(
            os.path.join(self.amadeus_root, "modules_embedding.pickle"), "rb"
        ) as f:
            database = pickle.load(f)

        result = []

        for k, v in database.items():
            distance = cosine_similarity([query_embedding], [v])
            result.append((k, distance))

        # sorted in descending order
        sorted_results = sorted(result, key=lambda x: x[1])[::-1]
        return sorted_results
