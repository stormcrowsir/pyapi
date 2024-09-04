import torch.nn.functional as F
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM

model_id = "mistral.mistral-7b-instruct-v0:2"

def extract_info(texts):
    template = """Given job post below, extract both implicit and explicit information about required technicall skill to have, make it specific into a list :\n\n  {question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm = HuggingFaceEndpoint(
    #         repo_id=repo_id, max_length=512
    #         # , temperature=0.5
    #     )
    llm = BedrockLLM(
        model_id=model_id, region_name='ap-southeast-2'
    )
    qaa = LLMChain(prompt=prompt, llm=llm)
    result = qaa.run(str(texts))

    return str(result)
