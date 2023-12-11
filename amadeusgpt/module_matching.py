import glob
import os
import pickle
import time
import openai
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

amadeus_root = os.path.dirname(os.path.realpath(__file__))
openai.organization = ""
if "streamlit_app" in os.environ:
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        st.session_state["exist_valid_openai_api_key"] = True
    else:
        openai.api_key = st.session_state.get("OPENAI_API_KEY", "")


def save_embeddings():
    result = {}

    module_apis = glob.glob(os.path.join(amadeus_root, "modules/api/*/*.py"))

    for module in module_apis:
        with open(module, "r") as f:
            text = f.read()
            model = "text-embedding-ada-002"
            try:
                embedding = openai.Embedding.create(input=[text], model=model)["data"][
                    0
                ]["embedding"]
                result[module] = embedding
            except openai.error.RateLimitError:
                st.error(
                    "It appears you are out of funds/free tokens from openAI - please check your account settings"
                )
                return None
            except:
                print(f"{module} is invalid")

    with open(os.path.join(amadeus_root, "modules_embedding.pickle"), "wb") as file:
        pickle.dump(result, file)


def match_module(query):
    model = "text-embedding-ada-002"
    openai.Model.list()
    success = False
    while not success:
        try:
            query_embedding = openai.Embedding.create(input=[query], model=model)[
                "data"
            ][0]["embedding"]
            success = True
        except openai.error.RateLimitError:
            st.error(
                "It appears you are out of funds/free tokens from openAI - please check your account settings"
            )
            return None
        except:
            print("something is wrong with embedding API. Reconnecting")
            time.sleep(3)

    if not os.path.exists(os.path.join(amadeus_root, "modules_embedding.pickle")):
        save_embeddings()

    with open(os.path.join(amadeus_root, "modules_embedding.pickle"), "rb") as f:
        database = pickle.load(f)

    result = []

    for k, v in database.items():
        distance = cosine_similarity([query_embedding], [v])
        result.append((k, distance))

    # sorted in descending order
    sorted_results = sorted(result, key=lambda x: x[1])[::-1]
    return sorted_results


if __name__ == "__main__":
    results = match_module("load monkey dataset")
