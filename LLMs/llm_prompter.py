# %%
import os
os.environ["OPENAI_API_BASE"] = 'http://localhost:8000/v1'
os.environ["OPENAI_API_KEY"] = 'EMPTY'
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("state_of_the_union.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="gpt-3.5-turbo")

questions = [
    "Who is the speaker",
    "What did the president say about Ketanji Brown Jackson",
    "What are the threats to America",
    "Who are mentioned in the speech",
    "Who is the vice president",
    "How many projects were announced",
]

for query in questions:
    print("Query:", query)
    print("Answer:", index.query(query, llm=llm))
# %%
from typing import List

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

class Reservation(BaseModel):
    date: str = Field(description="reservation date")
    time: str = Field(description="reservation time")
    party_size: int = Field(description="number of people")
    cuisine: str = Field(description="preferred cuisine")

parser = PydanticOutputParser(pydantic_object=Reservation)

reservation_template = '''
  Book us a nice table for two this Friday at 6:00 PM. 
  Choose any cuisine, it doesn't matter. Send the confirmation by email.

  Our location is: {query}

  Format instructions:
  {format_instructions}
'''

prompt = PromptTemplate(
    template=reservation_template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
) 

_input = prompt.format_prompt(query="San Francisco, CA")

output = model(_input.to_string())

reservation = parser.parse(output)

print(_input.to_string())

for parameter in reservation.__fields__:
    print(f"{parameter}: {reservation.__dict__[parameter]},  {type(reservation.__dict__[parameter])}")
# %%
print(output)
# %%
test_prompt = {"id": "arxiv_0", "graph": {"node_idx": 0, "edge_index": [[1, 2, 3, 4], [0, 0, 1, 1]], "node_list": [0, 52893, 93487, 14528, 71730], "node_label": ["Cryptography and Security", "Machine Learning", "Machine Learning", "Machine Learning", "Machine Learning"]}, "conversations": {"from": "human", "value": "Given a citation graph: \n<graph>\n, where the 0th node is the target paper, with the following information: \nTitle: evasion attacks against machine learning at test time\nAbstract: In security-sensitive applications, the success of machine learning depends on a thorough vetting of their resistance to adversarial data. In one pertinent, well-motivated attack scenario, an adversary may attempt to evade a deployed system at test time by carefully manipulating attack samples. In this work, we present a simple but effective gradient-based approach that can be exploited to systematically assess the security of several, widely-used classification algorithms against evasion attacks. Following a recently proposed framework for security evaluation, we simulate attack scenarios that exhibit different risk levels for the classifier by increasing the attacker's knowledge of the system and her ability to manipulate attack samples. This gives the classifier designer a better picture of the classifier performance under evasion attacks, and allows him to perform a more informed model selection (or parameter setting). We evaluate our approach on the relevant security task of malware detection in PDF files, and show that such systems can be easily evaded. We also sketch some countermeasures suggested by our analysis.\nAnd in the 'node label' list, you'll find subcategories corresponding to the 2-hop neighbors of the target paper as per the 'node_list.Question: Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form \"cs.XX\" with full name of the category."}}

text = test_prompt['conversations']['value']


start_index = text.find("Title") + len("Title") 
end_index = text.find("And") 

if start_index != -1 and end_index != -1:  # 确保两个词都被找到
    result = 'Title' + text[start_index:end_index].strip()  # 使用切片获取两者之间的内容，并去除可能存在的空白
    print(result)
else:
    print("error")
# %%
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
os.environ["OPENAI_API_BASE"] = 'http://localhost:8000/v1'
os.environ["OPENAI_API_KEY"] = 'EMPTY'
model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

class Classification(BaseModel):
    category: str = Field(description="the most likely arXiv CS sub-categories of this paper directly, in the form \"cs.XX\" with full name of the category.")
    explanation: str = Field(description="explanation for the classification")


parser = PydanticOutputParser(pydantic_object=Classification)


classification_template = '''
    Given a citation graph: 
    \n<graph>\n{structure_input}
    , where the 0th node is the target paper, with the following information: 
    {text_input}
    And in the 'node label' list, you'll find subcategories corresponding to the 2-hop neighbors of the target paper as per the 'node_list'.
    
    Format instructions:
    {format_instructions}
    '''

prompt = PromptTemplate(
    template=classification_template,
    input_variables=["structure_input", "text_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
) 

_input = prompt.format_prompt(structure_input=test_prompt['graph'], text_input=result)
print(_input)
output = model(_input.to_string())
print(output)
# %%
classification = parser.parse(output)

print(_input.to_string())

for parameter in classification.__fields__:
    print(f"{parameter}: {classification.__dict__[parameter]},  {type(classification.__dict__[parameter])}")
#
# %%
chain = prompt | model | parser
chain.invoke({"structure_input": test_prompt['graph'], "text_input": result})

# %%


from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import os
os.environ["OPENAI_API_BASE"] = 'https://openai-proxy-xuehongyanl.wanderful.space/v1'
os.environ["OPENAI_API_KEY"] = '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'

model_name = 'gpt-3.5-turbo'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
    # You can add custom validation logic easily with Pydantic.
    @validator('setup')
    def question_ends_with_question_mark(cls, field):
        if field[-1] != '?':
            raise ValueError("Badly formed question!")
        return field

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."
_input = prompt.format_prompt(query=joke_query)

chain = prompt | model | parser
chain.invoke({"query": joke_query})

# %%
print(output)
# %%
print(_input.to_string())
# %%
parser.parse(output.lower())
# %%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ["OPENAI_API_BASE"] = 'https://openai-proxy-xuehongyanl.wanderful.space/v1'
os.environ["OPENAI_API_KEY"] = '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'
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})
# %%
'''llmchain 用法'''
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Customer: {question}
Assistant:""")

model_path = 'lmsys/vicuna-7b-v1.5'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=64, 
    temperature=0
)

llm = HuggingFacePipeline(pipeline=pipe)
chain = LLMChain(llm=llm, prompt=prompt)
res = chain.run(question="你好")
# %%
# agent exp

# LangChain相关模块的导入
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
# 加载个人的OpenAI Token
import os
os.environ["OPENAI_API_BASE"] = 'https://openai-proxy-xuehongyanl.wanderful.space/v1'
os.environ["OPENAI_API_KEY"] = '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'
key = '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'
# 创建OpenAI调用实例
# 在本示例中，大模型需要作为一个严谨的逻辑解析工具，所以temperature设置为0
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=key)
# 需要安装依赖库 pip install wikipedia# 加载内置工具 llm-math 和 wikipedia
tools = load_tools(["llm-math", "wikipedia"], llm=llm)
# 创建Agent实例
agent = initialize_agent(
    # 设置可以使用的工具
    tools=tools,
    # 设置逻辑解析所使用的大模型实例
    llm=llm,
    # 设置agent类型，CHAT表示agent使用了chat大模型，REACT表示在prompt生成时使用更合乎逻辑的方式，获取更严谨的结果
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # 设置agent可以自动处理解析失败的情况，由于使用大模型处理逻辑，返回值并不会100%严谨，所以可能会出现返回的额数据不符合解析的格式，导致解析失败
    # agent可以自动处理解析失败的情况，再次发送请求以期望获取到能正常解析的返回数据
    handle_parsing_errors=True,
    # 打开详细模式，会将agent调用细节输出到控制台
    verbose=True)
res = agent("1000的35%是多少？")
# %%

# LangChain相关模块的导入
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import tool

# 加载个人的OpenAI Token
key = 'open_ai_key'

# 创建OpenAI调用实例
# 在本示例中，大模型需要作为一个严谨的逻辑解析工具，所以temperature设置为0
llm = ChatOpenAI(temperature=0, openai_api_key=key)

# 使用tool装饰器，可以直接将一个方法转为BaseTool实例，用于放到Agent的tools参数中支持调用
# 编写一个工具，用于将中文文本转换成拼音，以列表的形式返回文本中的每一个字
@tool
def chinese_to_pinyin(query: str) -> str:
    # 方法名作为自定义tool的实例名称
    # query参数是经过大模型分析之后，送入当前tool的文本信息
    # 方法中必须要存在doc,这个doc会被作为tool的描述信息，提交给大模型用于判断什么时候怎么调用当前tool
    """接收中文文本，返回对应中文的拼音列表，能够将中文转换成拼音的工具，必须要接收一个中文文本作为输入参数，并且返回的时候总是一个列表数据"""
    # 使用pypinyin库将中文文本转换成拼音
    # 需要事先安装依赖库 pip3 install pypinyin
    from pypinyin import pinyin
    res = pinyin(query)
    # 将转换结果的格式修正一下[["zhong"],["wen"]] => "['zhong','wen']"
    return str([i[0] for i in res])


agent = initialize_agent(
    # 设置可以使用的工具列表
    tools=[chinese_to_pinyin],
    # 设置逻辑解析所使用的大模型实例
    llm=llm,
    # 打开详细模式，会将agent调用细节输出到控制台
    verbose=True)
# 待翻译文本
chinese_str = "打开Debug模式，将会输出更详细的日志信息，方便了解整体运行逻辑"
# 打开Debug模式，将会输出更详细的日志信息，方便了解整体运行逻辑
import langchain

langchain.debug = True
# 组装成最终的命令字符串，并调用agent运行解析，获取结果
command_str = f"将以下文本转换成拼音: \n{chinese_str}"
res = agent.run(command_str)
# 打印结果
print(res)
# 关闭Debug模式
langchain.debug = False