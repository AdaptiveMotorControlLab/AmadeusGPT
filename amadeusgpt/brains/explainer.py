from amadeusgpt.brains.base import BaseBrain
from amadeusgpt.system_prompts.explainer import _get_system_prompt
from amadeusgpt.utils import timer_decorator


class ExplainerBrain(BaseBrain):
    @classmethod
    def get_system_prompt(
        cls, user_input, thought_process, answer
    ):
        return _get_system_prompt(
            user_input, thought_process, answer
        )

    @classmethod
    @timer_decorator
    def generate_explanation(
            cls,
            user_input,
            thought_process,
            answer,
            plots
    ):
        """
        Explain to users how the solution came up
        """

        captions = ''
        if isinstance(plots, list):
            for plot in plots:
                if plot.plot_caption !='':
                    captions+=plot.plot_caption
            if captions!='':
                answer+=captions
                
        messages = [
            {
                "role": "system",
                "content": cls.get_system_prompt(
                    user_input,
                    thought_process,
                    answer,
                ),
            }
        ]
        response = cls.connect_gpt(messages, max_tokens=500)["choices"][0]["message"][
            "content"
        ]
        return response
