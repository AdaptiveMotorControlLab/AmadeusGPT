def _get_system_prompt():
    system_prompt = """ Your job is to correct a code that raised errors. You will be given the user query the code was written for, the code, the api docs and the error message. 
        The code must follow the api docs and error is likely caused by wrong use of api docs. DO NOT attempt to correct the code if the error asks you not to
    """
    system_prompt += "There are empirical rules that can help you debug:\n"
    system_prompt += f"Rule 1: If the code used a bodypart that does not exist, replace it with one that is semantically close from supported bodyparts. \n"
    return system_prompt
