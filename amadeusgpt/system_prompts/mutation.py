from os import system


def _get_system_prompt(core_api_docs, 
                       task_program_docs, 
                       task_program_to_be_mutated
                       ):
    system_prompt = f"""
You are an expert in both animal behavior and you understand how to write code. Your job is to write code that creates a new behavior from an existing one.

You will be provided with information that are organized in following blocks:
coreapidocs: this block contains information about the core apis that can help capture behaviors. They do not contain implementation details but they give you ideas what behaviors you can capture.
taskprograms: this block contains the description of the behaivors we alreay know. Make sure you don't repeat the same behavior.
task_program_to_be_mutated: this block contains the description of the behavior that you need to mutate.
Following are information you have. 
{core_api_docs}\n{task_program_docs}\n{task_program_to_be_mutated}\n

You need to justify why that is nontrivial change and why it's realistic using the coreapidocs.
Following are rules you should also follow:
1) Don't apply filter such as min_window or smooth_window

"""
    return system_prompt