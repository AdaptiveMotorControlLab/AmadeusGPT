def _get_system_prompt(
    task_description, function_code, interface_str, traceback_output
):
    system_prompt = """You will be forwarded a problematic code and helpful information. These are:  user query followed by <query:>, the api docs followed by <api_doc:>  and the code  followed by <func_str:> and the runtime error is followed by <errors:>. \n
    You need to write high level analysis about why the code raises runtime errors and write high level suggestions to revise the code. Make sure the code still satisfy user query and api docs. """

    return system_prompt
