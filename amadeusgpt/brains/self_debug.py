from amadeusgpt.brains.base import BaseBrain
from amadeusgpt.system_prompts.self_debug import _get_system_prompt
from amadeusgpt.logger import AmadeusLogger


class SelfDebugBrain(BaseBrain):
    @classmethod
    def get_system_prompt(cls):
        return _get_system_prompt()

    @classmethod
    def debug_and_retry(cls, 
                        user_query = "",
                        error_code = "",
                        diagnosis = "",
                        api_docs = "", 
                        error_message = ""):
        #AmadeusLogger.info(f"the diagnosis was: {diagnosis}")
        #AmadeusLogger.info(f"the traceback used for error correction {error_message}")

        system_prompt = cls.get_system_prompt()

        user_prompt = f"""
        <user_query> {user_query} </user_query>
        <error_code> {error_code} </error_code>
        <api_docs> {api_docs} </api_docs>
        <error_message> {error_message} </error_message>
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
 

        response = cls.connect_gpt(messages, max_tokens=500)
        text, function_codes, thought_process = cls.parse_openai_response(response)
        return text, function_codes, thought_process
