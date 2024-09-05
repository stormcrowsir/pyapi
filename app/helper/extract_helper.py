from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM

model_id = "mistral.mistral-7b-instruct-v0:2"

def extract_info(texts):
    template = """You are an expert at professional field and education, given list of possible technical skill below:

    Language :
    JavaScript, HTML/CSS, SQL, Python, TypeScript, Java, C#, C++, PHP, C, Go, Rust, Kotlin, Ruby, Dart, Lua, Swift, Visual Basic (.Net), 

    Database :
    PostgreSQL, MySQL, SQLite, MongoDB, Microsoft SQL Server, Redis, MariaDB, Elasticsearch, Oracle, DynamoDB, Firebase Realtime Database, Cloud Firestore, BigQuery, Microsoft Access, H2, Cosmos DB, Supabase, InfluxDB, Cassandra, Snowflake, Neo4J

    Cloud Provider :
    Amazon Web Services (AWS), Microsoft Azure, Google Cloud, Firebase, Cloudflare, Digital Ocean, Heroku, Vercel, Netlify, VMware,

    Framework :
    React, Node.js, jQuery, Angular, Express, ASP.NET Core, Vue.js, Next.js, ASP.NET, Spring Boot, WordPress, Flask, Django, Laravel, AngularJS, FastAPI, Ruby on Rails, Svelte, NestJS, Blazor, Nuxt.js, Symfony, Gatsby, Phoenix, Fastify, Deno, CodeIgniter

    Other tools :
    Docker, npm, Pip, Homebrew, Yarn, Webpack, Make, Kubernetes, NuGet, Maven (build tool), Gradle, Vite, Visual Studio Solution, CMake, Cargo, GNU GCC, Terraform

    Based on the previous given list of technicall skill pick 1 or 2 in each category which corespond to the job post given : {question}
    
    Please answer in this format [Number]. [Skill Name]: [Reason why need that skill in one sentence], as a list only maximum of 20 items, do not repeat yourself."""
    
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
