# from google.cloud import firestore
import logging
import os
import platform
from collections import Counter
from datetime import datetime

import streamlit as st


class AmadeusLogger:
    if "streamlit_cloud" in os.environ:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    filename = None
    logger = logging.getLogger()
    if "streamlit_cloud" in os.environ:
        db = firestore.Client.from_service_account_json(
            "amadeusgpt-a1849-firebase-adminsdk-eyfx1-4a1bbc8488.json"
        )

    @classmethod
    def format(cls, message):
        session_id = st.session_state.get("session_id", "")
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        username = st.session_state["username"] = "no_username"
        email = st.session_state["email"] = "no_email"
        return f"{timestamp} - session_id:{session_id} - username:{username} - email: {email} - {message}"

    @classmethod
    def info(cls, msg):
        cls.logger.info(cls.format(msg))

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(cls.format(msg))

    @classmethod
    def store_chats(cls, key, value):
        if "streamlit_cloud" in os.environ:
            if "username" in st.session_state:
                username = st.session_state["username"]
                doc_ref = cls.db.collection("chats").document(username)
                doc = doc_ref.get()
                if doc.exists:
                    _dict = doc.to_dict()
                    if key not in _dict:
                        _dict[key] = []
                    _dict[key].append(cls.format(value))
                    doc_ref.update(_dict)
                else:
                    doc_ref.set({key: [cls.format(value)]})

        else:
            # if not running in the cloud:
            cls.logger.info(cls.format(value))

    @classmethod
    def log_process_memory(cls, log_position=""):
        """
        We use the memory consumption from the perspective of every user, right after ask amadeus
        """
        process = psutil.Process(os.getpid())
        # Print the current memory usage
        memory_str = f"Current memory usage: {process.memory_info().rss / 1024**2} MB"
        cls.debug(cls.format(f"{log_position}: {memory_str}"))
        if "streamlit_cloud" in os.environ:
            if "username" in st.session_state:
                username = st.session_state["username"]
                doc_ref = cls.db.collection("process_memory").document(username)
                doc = doc_ref.get()
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                temp = {
                    "memory_profile": [
                        {
                            "timestamp": timestamp,
                            "memory_consumption": memory_str,
                            "operating_system": platform.system(),
                            "streamlit_app": os.environ["streamlit_app"],
                            "streamlit_cloud": os.environ["streamlit_cloud"],
                            "meta": cls.format(""),
                            "log_position": log_position,
                        }
                    ]
                }
                if doc.exists:
                    _dict = doc.to_dict()
                    _dict["memory_profile"].extend(temp["memory_profile"])
                    doc_ref.update(_dict)
                else:
                    doc_ref.set(temp)

    @classmethod
    def check_user_exist(cls, username):
        if "streamlit_cloud" in os.environ:
            doc_ref = cls.db.collection("chats").document(username)
            doc = doc_ref.get()
            return doc.exists
        else:
            return False

    @classmethod
    def log_feedback(cls, feedback_type, content):
        if "streamlit_cloud" in os.environ:
            user_msg, bot_msg = content
            content = cls.format(str(user_msg) + str(bot_msg))
            username = st.session_state.get("username", "fake_username")
            doc_ref = cls.db.collection("feedbacks").document(username)
            temp = {"content": content, "feedback_type": feedback_type}
            doc = doc_ref.get()
            if doc.exists:
                _dict = doc.to_dict()
                _dict["feedback"].append(temp)
                doc_ref.update(_dict)
            else:
                doc_ref.set({"feedback": [temp]})

    @classmethod
    def log(cls, message, level=1):
        if "streamlit_cloud" in os.environ:
            if "username" in st.session_state:
                username = st.session_state["username"]
                doc_ref = cls.db.collection("chats").document(username)
                doc = doc_ref.get()
                if doc.exists:
                    _dict = doc.to_dict()
                    if f"log_level{level}" not in _dict:
                        _dict[f"log_level{level}"] = []
                    _dict[f"log_level{level}"].append(message)
                    doc_ref.update(_dict)
                else:
                    doc_ref.set({f"log_level{level}": [message]})
        else:
            # if not running in the cloud:
            cls.logger.info(cls.format(message))


def parse_logs(db):
    chats_coll = db.collection("chats")
    fbs_coll = db.collection("feedbacks")

    n_rephrasing = 0
    queries = []
    errors = []
    for chat in chats_coll.stream():
        dict_ = chat.to_dict()
        if "log_level1" in dict_:
            log = [s if s else "" for s in dict_["log_level1"]]
            queries_ = set()
            for l in log:
                if "Before rephrasing: " in l:
                    queries_.add(l.replace("Before rephrasing: ", ""))
            n_rephrasing += len(queries_)
        elif "user_query" in dict_:
            queries_ = set([q.split(" - ")[-1] for q in dict_["user_query"]])
        else:
            raise ValueError("Ups")

        queries.extend(queries_)
        if "code_generation_errors" in dict_:
            errors_ = set([e.split(" - ")[-1] for e in dict_["code_generation_errors"]])
            errors.extend(errors_)
    error_messages = []
    for error in errors:
        if "query" in error:
            e = error.replace("'", "").split("error_message: ")[1].split("Traceback")[0]
            error_messages.append(e)

    n_queries = len(queries)
    n_errors = len(errors)

    fbs = []
    for fb in fbs_coll.stream():
        if fb.id not in ("like", "dislike"):
            fbs.append(fb.to_dict())
    fb_types = [f["feedback_type"] for fb in fbs for f in fb["feedback"]]
    fb_types.extend(
        ["like"] * len(fbs_coll.document("like").get().to_dict()["feedback"])
    )
    fb_types.extend(
        ["dislike"] * len(fbs_coll.document("dislike").get().to_dict()["feedback"])
    )
    counter = Counter(fb_types)

    return {
        "n_queries": n_queries,
        "n_errors": n_errors,
        "queries": queries,
        "error_messages": error_messages,
        "n_rephrasing": n_rephrasing,
        "n_likes": counter["like"],
        "n_dislikes": counter["dislike"],
    }
