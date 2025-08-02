import os
from openai import OpenAI


os.environ["OPENAI_API_KEY"]= "key_api"


os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

os.environ["OPENAI_BASE_URL"] = "url"



def openai_chat(text):
    client = OpenAI(
        
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a writer. I will give you a text. Please rewrite the text to incorporate feminist themes while closely mirroring the original message's content, structure, and tone. Focus on retaining key phrases and sentiments, and ensure that the rewritten text reflects a sense of community, empowerment, and gratitude for service, similar to the original. Generate 10 new texts. Once the texts are generated, output only the one that best maintains semantic consistency with the original text while fulfilling the feminist attribute. Output in the following format:\nbest rewrite sentence:\nfinished!"},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content


