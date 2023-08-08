import os 
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
import pandas as pd
from langchain.chat_models import ChatOpenAI        
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import openai
import gradio as gr
import re
import json
import requests
import pandas as pd
import math
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain


os.environ['OPENAI_API_TYPE'] = "azure"
# The API version you want to use: set this to `2022-12-01` for the released version.
os.environ['OPENAI_API_VERSION'] = "2022-12-01"
# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
# os.environ['OPENAI_API_BASE']="https://pz-ew-aoi-np-digitaltrans-001.openai.azure.com/"
os.environ['OPENAI_API_BASE'] = "https://pz-ew-aoi-np-digitaltrans-002.openai.azure.com/"
# The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
# os.environ['OPENAI_API_KEY']="65f234dbba4b4264b734715a74bcf03d"
os.environ['OPENAI_API_KEY']="d4d01a5d410a4c69823b62d0cc205ea5"

os.environ["LANGCHAIN_HANDLER"] = "langchain"



BASE_URL = "https://pz-ew-aoi-np-digitaltrans-002.openai.azure.com/"
API_KEY = "d4d01a5d410a4c69823b62d0cc205ea5"
DEPLOYMENT_NAME = "gpt-35-turbo-16k"
DEPLOYMENT_NAME_GPT4 = 'gpt-4'
DEPLOYMENT_NAME_GPT432k = 'gpt-4-32k'
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
            openai_api_version="2023-07-01-preview",
            deployment_name=DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            temperature=0,max_tokens=500
        )

llm_4 = AzureChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_version="2023-07-01-preview",
            deployment_name=DEPLOYMENT_NAME_GPT4,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            temperature=0,max_tokens=500
)
llm_432k = AzureChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_version="2023-07-01-preview",
            deployment_name=DEPLOYMENT_NAME_GPT432k,
            openai_api_key=API_KEY,
            openai_api_type = "azure",
            temperature=0,max_tokens=500
)



def getSentiment():
    data = pd.read_csv("data.csv")
    data = data[data.comment.notna()]
    # split = RecursiveCharacterTextSplitter(chunk_size=7000,chunk_overlap=0)
    split = TokenTextSplitter(chunk_size=30000,chunk_overlap=5)
    docs = split.split_text("".join(list(data['comment'])))
    
    doc1 = [Document(page_content = d,metadata={'source':'local'}) for d in docs]
    templateSent = """Please act as a machine learning model trained for perform a supervised learning task, 
        for evaluation the sentiment of reviews from list of reviews from {context_str} and 


        The value of sentiment must be "positive"  or "negative"

        evaluate the count of Positive sentiment of a reviews from list of comments.
        The value of Positive field must be number

        evaluate the count of Negative sentiment of a reviews from list of comments 
        The value of Negative field must be number

        only give your answer in Dictionary file format as below format inside curly braces.

        
        example:
        sentiment : Sentiment , 
        positive : positive , 
        negative : negative 
        
        {question}
        
    """
        #example:
        #sentiment : $Sentiment$  
        #positive : $positive$
        #negative : $negative$

        #extract the Positive sentiment commnets and count it.

        #extract the Negative sentiment comments and count it
    prompt = PromptTemplate(template=templateSent, input_variables=["context_str","question"])

    chain = load_qa_chain(llm_432k, chain_type="refine",question_prompt = prompt,return_refine_steps=True)
    query = "Tell me overall sentiment, number of positive sentiment and number of negative sentiment?"
    out = chain({"input_documents":doc1,"question":query})
    return(out['output_text'])  #(out['intermediate_steps'])  #
    

    
#jcomments=[]
def getReviews(URL):
    url = URL
    r = re.search(r"i\.(\d+)\.(\d+)", url)
    shop_id, item_id = r[1], r[2]
    ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

    offset = 0
    d = {"username": [], "rating": [], "comment": []}
    while True:
        data = requests.get(
            ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        #data = [x for x in data if x is not None]

        # uncomment this to print all data:
        # print(json.dumps(data, indent=4))
        
        i = 1
        
        if data["data"]["ratings"] is not None:
            for i, rating in enumerate(data["data"]["ratings"], 1):
                if rating is not None:
                    d["username"].append(rating["author_username"])
                    d["rating"].append(rating["rating_star"])
                    d["comment"].append(rating["comment"])

                    print(rating["author_username"])
                    print(rating["rating_star"])
                    print(rating["comment"])
                    print("-" * 100)
            

        if i % 20:
            break

        offset += 20
    
    df = pd.DataFrame(d)
    df=df[df.comment.notna()]
    df.to_csv("data.csv", index=False)
    df = pd.read_csv('data.csv') 
    comments=df.comment
    ratings=df.rating
    jcomments = json.dumps(comments.tolist())
    response=getSentiment()
    resp = response.replace("{",'').replace("}",'')
    out = response.replace("\n",'')
    try:
        dctReview=json.loads(response)
        if 'Positive' in dctReview.keys():
            pos=int(dctReview['Positive'])
            neg=int(dctReview['Negative'])
        else:
            pos=int(dctReview['positive'])
            neg=int(dctReview['negative'])
    except:
        pos=0
        neg=0
        
    simple = pd.DataFrame({
    'Positive & Negative Sentiments': ['Positive', 'Negative'],
    'Number of comments': [pos, neg]
       })
    
    return resp,gr.BarPlot.update(
           simple,
           x="Positive & Negative Sentiments",
           y="Number of comments",
           title="",
           #tooltip=['a', 'b'],
           y_lim=[0, pos+20]
       )
    
    
    

def CustomChatGPT(user_input):
    data = pd.read_csv("data.csv")
    data = data[data.comment.notna()]
    # split = RecursiveCharacterTextSplitter(chunk_size=7000,chunk_overlap=0)
    split = TokenTextSplitter(chunk_size=30000,chunk_overlap=5)
    docs = split.split_text("".join(list(data['comment'])))
    
    doc1 = [Document(page_content = d,metadata={'source':'local'}) for d in docs]
    templateQue = """You are an advanced virtual assistant, who can help in analyzing customer sentiment from a list of reviews.
        Each item in the below list is one customer review.
        Read the reviews and answer the following question as clearly and succintly as possible in english.
        
        {context_str}
        Example:
        question :  How many people like the quality of the product?
        answer : According to the reviews, 15 people are happy with the quality of the product
        question : What aspects of the product are people happy about?
        answer : People generally find the product easy to use and effective
        
        {question} 
        
    """

        #extract the Positive sentiment commnets and count it.

        #extract the Negative sentiment comments and count it
    prompt = PromptTemplate(template=templateQue, input_variables=["context_str","question"])

    chain = load_qa_chain(llm_432k, chain_type="refine",question_prompt = prompt,return_refine_steps=True)
    query= user_input
    #query = "Tell me overall sentiment, number of positive sentiment and number of negative sentiment?"
    out = chain({"input_documents":doc1,"question":query})
    #print(out['output_text'])  
    #print(out['intermediate_steps'])
    return(out['output_text'])  #(out['intermediate_steps'])  #


def main():
    with gr.Blocks(title = "Sentiment anlysis") as demo:
        with gr.Row():
                text1URL = gr.Textbox(label="URL",min_width=500)
                text2URL = gr.Textbox(label="Overall Sentiment",min_width=500)
                #Positive = gr.Number(value=pos, label="positive")
                #Negative = gr.Number(value=neg, label="negative")
                #boroughs = gr.CheckboxGroup(choices=["Queens", "Brooklyn", "Manhattan", "Bronx", "Staten Island"], value=["Queens", "Brooklyn"], label="Select Boroughs:")
                #btn = gr.Button(value="Update Filter")
                #map = gr.Plot()
                #plot= gr.Plot(16,0)
                #gr.BarPlot()
                plot = gr.BarPlot(label = "Graphical Representation",min_width=100) #show_label=False
        
        
        #with gr.Column():
                #plot = gr.BarPlot(show_label=False)
                
        with gr.Row():
                button1 = gr.Button("Submit")
                button1.click(getReviews,inputs=text1URL,outputs=[text2URL,plot]) #outputs=[text2URL,plot]
                #button1.click(getgraph,inputs=,outputs=)

        with gr.Row():
                text1QA = gr.Textbox(label="Ask Question")
                text2QA = gr.Textbox(label="Answer")

        with gr.Row():
                button1 = gr.Button("OK")
                button1.click(CustomChatGPT,text1QA,text2QA)

    demo.launch(server_name="0.0.0.0") #server_name="0.0.0.0"

if __name__ == "__main__":
    main()
