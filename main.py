
import openai


from ga_src.env import temperature, MAX_TOKEN, GENERATION, N_TRIALS, TP, QWEN, GPT, GEMINI, GLM, CLAUDE, GPT4o, apikeys, \
    base_urls
from ga_src.pipeline import get_prob_inst_set, run_single_trial, run_test, run_test_with_timeout, get_prob_test, \
    fun_search
from interaction import openai_code_evolve, fault_investigation
from MOTSP.motsp import MOTSP, TSP
# Set the model parameters



# def obtain_client(model_name):
#     if 'gpt' in model_name or 'gemini' in model_name or 'glm' in model_name:
#         return openai.OpenAI(api_key=apikeys[model_name], base_url=base_urls[model_name])
#     else:
#         return None
def code_evolving():
    RUNS = ['R1-', 'R2-', 'R3-']
    for model_name in [GPT]:
        for run in RUNS:
            client = openai.OpenAI(api_key=apikeys[model_name], base_url=base_urls[model_name])
            openai_code_evolve(
                client=client,
                model_name=model_name,
                temperature=temperature,
                max_token=MAX_TOKEN,
                problem=TP, # MOP for continuous / MOKP / MOTSP
                R=run
            )
    # investigate t on Gemini
    # for model_name in [GEMINI]:
    #     TS = ['T1-']
    #     for idx, temperatur in enumerate([0.5]):
    #         client = openai.OpenAI(api_key=apikeys[model_name], base_url=base_urls[model_name])
    #         openai_code_evolve(
    #             client=client,
    #             model_name=model_name,
    #             temperature=temperatur,
    #             max_token=MAX_TOKEN,
    #             problem=TP,  # MOP for continuous / MOKP / MOTSP
    #             R=TS[idx]
    #         )


if __name__ == '__main__':
    code_evolving()

# srXtbgNYvQ
# ps -def | grep main_GPT | grep -v grep
# nohup python main_GPT.py -u > nohup.out 2>&1 &

# ssh root@10.242.187.48 -p 22000