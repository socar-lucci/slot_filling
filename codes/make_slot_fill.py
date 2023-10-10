import pandas as pd
import openai
from tqdm import tqdm
from ast import literal_eval

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


def slot_fill(ontologies, message):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo-model",
        messages=[
            {
                "role": "system",
                "content": """
        [Role Explanation]
        1. You are an AI assistant to help solve customer's services.
        2. You are to help 상담사 in solving Customer's queries.
        3. You are ready to help extract Customer's urgent needs. 

[Instructions]
1. Read and understand the conversation between the 상담사 and 고객 in the provided examples.
2. Each case represents a conversation.
3. The Outputs of each Conversation are the keys and values that represent the core semantics of Customer's intent and circumstance.
4. The ontologies given under [Ontologies] are the possible slots for a conversation.
5. The ontologies are given in the form {Key: Values}.
6. Read and understand the Conversation given in Example 3.
7. Choose only the most relevent slots from the given [Ontologies] that are the most appropriate for the Customer's utterance. You must not take into consideration the utterance of Service Center.
8. If the possibe value is in the form of list, choose the possible value in the list and complete the ontology.
9. If the possible value is in the form of text string (i.e., 'str','price' etc), fill in the appropriate value to replace the text string. If you cannot fill in the values to substitute the text, do not output anything.
10. You must not output anything other than the words given in the [Ontologies] or the [Conversation].
11. Give me the output in the form of dict.
12. Do not output anything unless you are absolutely certain of your answer.
13. Do not create any words for the slots.

[Ontologies]
%s

[Example]
> Example 1.
Customer: 지정한 주차 장소에 가서 소카를 끌고 나오는데 주차요금을 지불했어요. 주차요금 돌려주시나요? \n
Service Center: 안녕하세요, 고객님. 쏘카 상담사 문도선 입니다. 정확한 상담을 위해 정보 확인 후 안내 도와드리도록 하겠습니다. 쏘카에 가입하신 지기천 고객님 본인 맞으십니까? \n 
Customer: 맞습니다
> Output 1.
{'본인확인': True, '주차비 개인부담여부': True}


> Example 2.
Customer: 안녕하세요 쏘카존인데 주차를 잘못해둬서 빼기 힘든데 어찌할까요 https://socar-cs.s3.ap-northeast-2.amazonaws.com/data/files/1/conversation/chat/323946bc-9b02-4294-91bb-334fd339fe08.jpg \n
Service Center: 윤윤설 상담사가 채팅에 참여하였습니다. 안녕하세요, 고객님. 쏘카 상담사 윤윤설 입니다. 무엇을 안내해드릴까요? 연결이 늦어 죄송합니다. 출차 불가 문의 주셨는데요, 정확한 확인을 위해 정보 확인 후 안내 도와드리도록 하겠습니다. 쏘카에 가입하신 정재우 고객님 본인 맞으십니까? \n
Customer: 아 네네
> Output 2.
{'본인확인': True, '증빙이미지 url': ['https://socar-cs.s3.ap-northeast-2.amazonaws.com/data/files/1/conversation/chat/323946bc-9b02-4294-91bb-334fd339fe08.jpg']}


> Example 3.
%s
> Output 3.
"""
                % (ontologies, message),
            }
        ],
        temperature=0,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.6,
        stop=None,
    )

    return response["choices"][0]["message"]["content"]


def main():
    speaker_dict = {"고객": "Customer", "상담사": "Agent"}
    df = pd.read_csv("../raw_dataset/dst_payment.csv")
    with open("../raw_dataset/paym_tmp_ko.txt", "r") as f:
        onts = f.readlines()
    ontologies = ",".join(onts)
    customers = []
    agents = []

    cnslt_ids = []
    slots = []
    out_df = pd.DataFrame(
        {
            "상담ID": cnslt_ids,
            "상담사 발화": agents,
            "고객 발화": customers,
            "고객 의도 / 상태": slots,
        }
    )
    out_df.to_csv("../raw_dataset/test_slot.csv", index=False)
    for idx in tqdm(range(len(df))):
        tmp = pd.read_csv("../raw_dataset/test_slot.csv")
        instance = df.iloc[idx]

        cnslt_id = instance["Conv_ID"]

        cnslt_ids.append(cnslt_id)

        a = literal_eval(instance["dialogue"])
        if len(a) < 2:
            if a[0]["role"] == "고객":
                customers.append("고객: " + a[0]["text"])
                agents.append("")
            elif a[0]["role"] == "상담사":
                agents.append("상담사: " + a[0]["text"])
                customers.append("")
        else:
            for utt in literal_eval(instance["dialogue"]):
                if utt["role"] == "고객":
                    customers.append("고객: " + utt["text"])
                elif utt["role"] == "상담사":
                    agents.append("상담사: " + utt["text"])

        txt_str = [
            speaker_dict[i["role"]] + ": " + i["text"]
            for i in literal_eval(df.iloc[idx]["dialogue"])
        ]

        response = slot_fill(",".join(onts[:-15]), "\n".join(txt_str))
        print(response)
        if response == None:
            break
        slots.append(response)

        tmp_df = pd.DataFrame(
            {
                "상담ID": cnslt_ids,
                "상담사 발화": agents,
                "고객 발화": customers,
                "고객 의도 / 상태": slots,
            }
        )
        # out_df = pd.read_csv("../raw_dataset/test_slot.csv")
        # out_df = pd.concat([out_df, tmp_df])
        tmp_df.to_csv("../raw_dataset/test_slot_tmp.csv", index=False)


if __name__ == "__main__":
    main()
