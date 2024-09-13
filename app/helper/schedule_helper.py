from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI



def summarize_schedule(schedule,question):
    
    template = """Given a schedule along with the tasks worked on that day : {schedule}
    
    Answer the given question based on the previous schedule : {query}
    
    Attach the prove from the schedule of the answer to back it up. Do not make up facts, answer with don't know if not enough data is provided.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["schedule","query"])
    # llm = HuggingFaceEndpoint(
    #         repo_id=repo_id, max_length=512
    #         # , temperature=0.5
    #     )
    llm = VertexAI(model_name="gemini-pro")
    qaa = LLMChain(prompt=prompt, llm=llm)
    result = qaa.run(schedule = str(schedule),query=str(question))

    return str(result)
