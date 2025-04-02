

OTHER = False
TP = 'MOP' # problems to solve
# llms
QWEN = 'qwen-max'
GPT = 'gpt-4-1106-preview'
GPT4o = 'gpt-4o'
GEMINI = 'api-gemini-1.5-pro'
GLM = 'glm-4-0520'
CLAUDE = 'claude-3-opus'
temps = {
    'MOP': 'multi-objective problems',
    'MOKP': 'multi-objective knapsack problems',
    'MOTSP': 'multi-objective traveling salesman problems'
}
run_times = {
    'MOP': 80,
    'MOKP': 80,
    'MOTSP': 80
}
apikeys = {
    GPT: 'sk-6qVWsbcMzwfkgBO11b1c5633D1664a2bB8Cf673eBa1696Df',
    GPT4o: 'sk-6qVWsbcMzwfkgBO11b1c5633D1664a2bB8Cf673eBa1696Df',
    GEMINI: 'sk-6qVWsbcMzwfkgBO11b1c5633D1664a2bB8Cf673eBa1696Df',
    QWEN: 'sk-27b248987223413d9eb63426bee37bd6',
    # GLM: 'e5043edd1502a94f27e1c9aa12bf249b.ImC4UYkz6fXFLObR',
    CLAUDE: 'sk-SMRNV23oIP9tSg5F44713a4f670b4211823b1a63465b07C2'
}
base_urls = {
    GPT: 'https://api.gptapi.us/v1/chat/completions',
    GPT4o: 'https://api.gptapi.us/v1/chat/completions',
    GEMINI: 'https://api.gptapi.us/v1/chat/completions',
    QWEN: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    # GLM: 'https://open.bigmodel.cn/api/paas/v4/',
    CLAUDE: 'https://api.gptapi.us/v1/chat/completions'
}
PROBLEM = temps[TP]
MAX_RUNNING_TIME = run_times[TP] # seconds
MODULE = 'rankings'
temperature = 0.5
MAX_TOKEN = 4000


GC_POP_SIZE = 10
GC_GENERATION = 10
MAX_MUTATION = 2
MAX_REPAIR = 1
APIUS_SLEEP = 1200
POP_SIZE = 100
MAX_ARCHIVE = 2 * POP_SIZE
N_TRIALS = 10
GENERATION = 200
N_THREAD = 10


# RUNS = [1, 2, 3, 4, 5]
# HRUNS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]