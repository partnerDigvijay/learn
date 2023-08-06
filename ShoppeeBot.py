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




def getSentiment():
         
        df = pd.read_csv('data.csv') 
        df=df[df.comment.notna()]
        comments=df.comment
        ratings=df.rating
        jcomments = json.dumps(comments.tolist())


        templateSent = """Please act as a machine learning model trained for perform a supervised learning task, 
        for extract the sentiment of reviews from list of Ratings from {ratings} and list of reviews from {text_review} and 

        Give your answer evaluating the sentiment field between the dollar sign, the value must be printed without dollar sign.
        The value of sentiment must be "positive"  or "negative"

        extract the count of Positive sentiment of a reviews from list of comments evaluating the Positive field between the dollar sign, the value must be printed without dollar sign.
        The value of Positive field must be number

        extract the count of Negative sentiment of a reviews from list of comments evaluating the Negative field between the dollar sign, the value must be printed without dollar sign.
        The value of Negative field must be number

        and give the answer in dictionary file

        example:
        Overall sentiment : $Sentiment$ 
        positive : $positive$
        negative : $negative$
        
"""
        #Example:
        #Overall sentiment : $sentiment$  , Positive comments : $Positive$, Negative comments: $Negative$ 
        #and Draw the barplot to depict number of Positive comments and Negative comments.


        #Give your answer considering ratings value >= 3 as "positive" sentiment and ratings value < 3 as "negative" sentiment
        #in dictionary file.
        
        #text_review ="Terima kasih paketnya sudah diterima dengan baik sesuai dengan pesanan"
        text_reviews=jcomments


        prompt = PromptTemplate(template=templateSent, input_variables=["text_review","ratings"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        
        response = llm_chain.run({"text_review": text_reviews,"ratings":ratings})
        
        return response

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
    dctReview=json.loads(out)
    if 'Positive' in dctReview.keys():
        pos=int(dctReview['Positive'])
        neg=int(dctReview['Negative'])
    else:
        pos=int(dctReview['positive'])
        neg=int(dctReview['negative'])
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
    
    df = pd.read_csv('data.csv') 
    df=df[df.comment.notna()]
    comments=df.comment
    ratings=df.rating
    jcomments = json.dumps(comments.tolist())
    
    templateQue = """You are an advanced virtual assistant, who can only help users answer their question of sentiment. 
        Extract the overall sentiments from list of Ratings from {ratings} and list of comments from {comments}.
        
        Give your answer  evaluating following lists {comments} and {ratings}.

        and Answer the following question refering above comments in English: {question} 
        
"""
        #Give your answer considering ratings value >= 3 as "positive" sentiment and ratings value < 3 as "negative" sentiment.
        #extract the Positive sentiment commnets and count it.

        #extract the Negative sentiment comments and count it
        
    prompt = PromptTemplate(template=templateQue, input_variables=["comments","ratings","question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question= user_input #How many comments are positive?    "What aspects are the users happy with?" #"Tell me the positive comments?" What is overall sentiment?
    Answer = llm_chain.run({"comments": jcomments,"ratings":ratings,"question":question})
    return Answer 




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
                    button1.click(getReviews,inputs=text1URL,outputs=[text2URL,plot])
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

