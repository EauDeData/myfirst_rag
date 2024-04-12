from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex, ServiceContext

import ctypes

from llama_cpp import llama_log_set

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en", device='cuda')

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096 * 2,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 40},
    verbose=False
)

Settings.llm = llm
Settings.embed_model = embed_model

loader = SimpleDirectoryReader(input_dir='./pdfs/', required_exts=['.pdf'], recursive=True)
docs = loader.load_data()

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(llm=llm, node_postprocessors=[LLMRerank()])


def silent_log_callback(level, message, user_data):
    pass


log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(silent_log_callback)
llama_log_set(log_callback, ctypes.c_void_p())

query_prompt = ("Context: {context_str}.\n"
                "-----------------------\n"
                "Please, answer the questions by writting 'Answer: <your_answer_here>'\n"
                "Query: {query_str}\n"
                "Answer:")
query_engine.update_prompts({'response_synthesizer:text_qa_template': PromptTemplate(query_prompt)})
qs = ["Can you summarize how "]
print('\n\n--------------QUERIES----------------')

max_tries = 10
for query in qs:
    print("\nQuestion:", query)
    for tries in range(max_tries):
        try:
            result = str(query_engine.query(query))
            assert not ('Empty Response' in result)
            print("Answer:", result)
            break
        except (IndexError, AssertionError):
            pass
    if tries == (max_tries - 1):
        print('(SCRIPT) Max tries on answering reached, skipping...')
