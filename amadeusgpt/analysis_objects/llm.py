from .base import SerializeableObject


class LLM:
    params = ['max_tokens',
    'gpt_model']

    dynamic_data = ['context_window',
    'history',
    'usage',
    'short_term_memory',
    'long_term_memory',
    'accumulated_tokens']

    