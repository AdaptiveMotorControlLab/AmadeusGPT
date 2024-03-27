import sys

def _get_system_prompt(parent1_docs, parent2_docs, composition_type):
    system_prompt = f"""You are an expert in animal behavior.
You will be given two description of behaviors and the way I want to combine them.
There are 3 ways to combine behaviors, they are "logical_and", "logical_or" and "sequential".
'logical_and': two behaviors happen at the same time.
'logical_or': one of the behaviors happen
'sequential': the second behavior happens after the first behavior.
Make sure you generate a new description that reflect that two behaviors are combined in the way I want.
Following are information you have.
description of behavior1: \n{parent1_docs}\n
description of behavior2: \n{parent2_docs}\n
Combination_method: \n{composition_type}\n
I want you to write a template function with name, description and with a template code that I will modify later.

The template should look like following:

def name_of_behavior(config):
    '''
    Description of the behavior.
    '''
    <template>


Make sure the style of the description should be similar to ones given. The name of the behavor, if multiple words, should be connected by underscores.




"""
    
    return system_prompt
