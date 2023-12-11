from amadeusgpt.brains.base import BaseBrain, classproperty
from amadeusgpt.system_prompts.rephraser import _get_system_prompt


class RephraserBrain(BaseBrain):
    @classmethod
    def get_system_prompt(cls):
        return _get_system_prompt()

    @classproperty
    def correction_k(cls):
        if not Database.exist(cls.__name__, "correction_k"):
            Database.add(cls.__name__, "correction_k", 1)
        return Database.get(cls.__name__, "correction_k")

    @classmethod
    def generate_iid(cls, user_input):
        """
        Try to ask the question like asked in API docs
        """
        messages = [
            {"role": "system", "content": cls.get_system_prompt()},
            {"role": "user", "content": f"{user_input}"},
        ]
        # use gpt 3.5 for rephraser to avoid using gpt-4 too fast to run into rate limit
        ret = cls.connect_gpt(messages, max_tokens=200, gpt_model="gpt-3.5-turbo")
        if ret is None:
            return None
        response = ret["choices"][0]["message"]["content"].strip()
        return response

    @classmethod
    def generate_equivalent(cls, user_input, k=5):
        """
        Generate equivalent messages based on base questions
        """

        messages = [
            {
                "role": "system",
                "content": f"Generate equivalent sentences with proper English using {k} different expression. \
                                                    Every generated sentence starts with <start> and end with </end>",
            },
            {"role": "user", "content": f"{user_input}"},
        ]

        AmadeusLogger.info("user input:")
        AmadeusLogger.info(user_input)
        response = cls.connect_gpt(messages)["choices"][0]["message"]["content"]
        AmadeusLogger.info("response:")
        AmadeusLogger.info(response)
        pattern = r"<start>\s*(.*?)\s*<\/end>"
        # parse the equivalent sentences
        matches = re.findall(pattern, response)
        sentences = [match.strip() for match in matches]

        return sentences
