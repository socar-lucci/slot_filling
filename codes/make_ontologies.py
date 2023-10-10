import openai
from ast import literal_eval
from tqdm import tqdm
import pandas as pd
import json
import os

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


def gpt_slot_response(ontologies, message):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo-model",
        messages="""

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
8. If the possible value is in the form of list, choose the possible value in the list and complete the ontology.
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
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop = ["\n"]
    )

    return response.choices[0].text.strip()


def make_ontology_azure(ontologies, conversation):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo-model",
        messages=[
            {
                "role": "user",
                "content": """
            [Instructions]
1. Read and understand the conversation between the customer (고객) and the service center (상담사) in the provided examples, under [Examples].
2. The objective is to discover key slot values necessary to track the state of the customer.
3. The ontologies given under [Ontologies] are a list of possible slots for a conversation.
4. The ontologies are given in the form {Slot Key: Possible Values/type}.
5. Try to choose the most relevant slots from the given [Ontologies] that are the most appropriate for the customer's utterance. 
6. If you can find relevant samples from the provided [Ontologies], you need to directly use them.
7. If cannot find relevant ontologies from [Ontologies], make the {Slot Key: Possible Values/ Types} on your own. 
8. The Slot Keys should be relevant to the customer's state
9. The Slot Key must not be a type of customer's intent, such as '주차비 관련 문의'.
10. Try to create general Possible Values for each Slot Keys only if you are absolutely sure.
11. If you are not absolutely sure, leave it as the type of the Possible Values such as boolean, str, int.
12. You must not add specific values for the Possible Values.
13. The Slot Key and the Possible Values should be concise, and based only on information from the conversation.
14. Please remove the slots that are not representative of the customer's state.
15. Remember, only the state and need of the customer are of interest.
16. Do not add the type of customer's intent.
17. Do not use information from the service center. 
18. Do not make up words unless you are absolutely certain of your answer.
19. If you cannot make an appropriate output, return an empty dict.
20. Solve the task in Example 4.

[Ontologies]
%s

[Example]
> Example 1.
Conversation: 
싱딤사: 문도선 상담사가 채팅에 참여하였습니다. 안녕하세요, 고객님. 쏘카 상담사 문도선 입니다. 무엇을 안내해드릴까요? 연결지연으로 불편드려 죄송합니다. 정확한 상담을 위해 정보 확인 후 안내 도와드리도록 하겠습니다. 쏘카에 가입하신 지기천 고객님 본인 맞으십니까? \n 고객: 맞습니다 \n 싱딤시: 소중한 정보 확인 감사합니다. 고객님 부름건 이용하신거 맞으실까요? \n 고객: 네

> Output 1.
{'본인 확인 결과': [None, True, False], '고객 직접 지불': [None, True, False], '부름 이용': [True, False, None]}


> Example 2.
Conversation:
상담사: 윤윤설 상담사가 채팅에 참여하였습니다. 안녕하세요, 고객님. 쏘카 상담사 윤윤설 입니다. 무엇을 안내해드릴까요? 연결이 늦어 죄송합니다. 출차 불가 문의 주셨는데요, 정확한 확인을 위해 정보 확인 후 안내 도와드리도록 하겠습니다. 쏘카에 가입하신 정재우 고객님 본인 맞으십니까? \n 고객: 아 네네

> Output 2.
{'본인 확인 결과': [None, True, False]}


> Example 3.
Conversation:
상담사 : 윤윤설 상담사가 채팅에 참여하였습니다. 안녕하세요, 고객님. 쏘카 상담사 윤윤설 입니다. 무엇을 안내해드릴까요? \n 고객: 안녕하세요 쏘카존인데 주차를 잘못해둬서 빼기 힘든데 어찌할까요 https://socar-cs.s3.ap-northeast-2.amazonaws.com/data/files/1/conversation/chat/323946bc-9b02-4294-91bb-334fd339fe08.jpg


> Output 3.
{'이미지 url': ['str']}

> Example 4.
Conversation:
%s

> Output 4.
"""
                % (ontologies, conversation),
            }
        ],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop = ["\n"]
    )

    # return response.choices[0]["message"]["content"]
    return response["choices"][0]["message"]["content"]


def main():
    dst_raw = pd.read_csv("../raw_dataset/dst_for_ont_refund.csv")
    ontologies = {}
    str_ont = []
    # for idx in range(5):
    # for idx in range(1):
    for idx in tqdm(range(len(dst_raw))):
        try:
            response = make_ontology_azure(
                str(ontologies), dst_raw.iloc[idx]["conversations"]
            )

            ontology = literal_eval(response)
            for key, value in ontology.items():
                if key not in ontologies.keys():
                    ontologies[key] = value

                    with open("../raw_dataset/refund_tmp.json", "w") as json_file:
                        json.dump(ontologies, json_file, ensure_ascii=False)

                print(str({key: value}))
        except:
            pass


if __name__ == "__main__":
    main()
