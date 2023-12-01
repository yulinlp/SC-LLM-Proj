import openai
import pandas as pd
import sys
from sklearn.metrics import f1_score
import time

log = open('chat.log', "a",encoding='utf-8')
sys.stdout = log
zouzou = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJ6eXgucmlja3l5QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLXprZHdDaGVrdEN4ZFhMOHV3bXRnTmJYTSJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjQ2ZjEzMjQzMzUxNDJhMjcxMDZjNzgxIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5NTYyNjk5OSwiZXhwIjoxNjk2ODM2NTk5LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIGVtYWlsIHByb2ZpbGUgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.VSwiMuZRdB8xv8cBfyaSe26FDxbtmpw9vB2l-N1zJYzpOPGn0a_6-se3Y93Gt9f_d-1Lnpg-PEK-XUy7Pakt0cL0o8uHr2h1TV4qXrTEK4hsJRRLJs1vV1CKatM8CWm9gvOKSa_6aVkcC_MPbyhMnqYxo5jHq8PFFfrD5Iyu_SYB6VK1QtjFU19gXA4GHkFtNWLSGPTuFDaDKMayJ9fo4EcGkiL-SQL4_XiyVzFBQebzMqvITkK_iYAQ0eXFkaWancFa_WNXHjK6aWa1f_Y3NFC_ig0NJRAsFF5xgjc0Yl61DT4bBaH5HjNH61XI-a4FaCJAtb6GWBraZirh6exejg'
lin = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiIxNTY1OTE4NDEyQHFxLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLWtYZUNZRUdEd2NkdUR1TWJFWGZ4ODI5dCJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjM5YjI0ZDU5Nzk5NmQ5ZmRmYjFkYmJjIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5Njc3NzI4MiwiZXhwIjoxNjk3NjQxMjgyLCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.m4yCClMj-mHWC15LMIPcSsQfzhiF2NiXEEQspIdMRctWPT7WRFRtC4yHXSaUWou-_DSz5fk6-7ZlEZdWo1gzWoFu76ixi1LkJchbgBKKqJ3VND2ww4egHqmkiZRTt6d_0ZbSNqvqKeLqU7IveU17c_U52SADlxTc5igUOd1bjBqosiC06g4AlW2ZaZVZxdlYgbqfT_T07jt-fz-Q1IgGxQMlH_3qCAp7gPGb64VHR28Bz3la1PKSrudjumpOoqvq8qROfm4e8FqKhBat7G7EB8pMs73n9MXJzcmCJx6bBCZb2xquMbPIHdYtjsePlrGDsN1FEgiQOzTTFks3JfVHAg'
hehezi = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtdXl1ZW5ob3NoaW5vQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLXE3NnlNN0YxMWU1WHZwTkhkTHEwMURaaiJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMDI1ODAyOTk1NTk1MzM2MjYwMTEiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNjk2NjAzNzk3LCJleHAiOjE2OTc0Njc3OTcsImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.r1WWgSCnuZloVpnGmNcEjIztx06zn2m3A9YfdR03uuiw3TZ7VaJLnwi6ctOp7zQUR-s1S3QUizwZgmcmp15rsxOGHI2NUKvPaOvQhML_s8uZXxoNsXVYGFWhG2vf33bna3_kKFps0UH4QSG5WYgFxXESnL7ml9hOn6h9maJALmMGRnUeoJ21hLe14Egae4b92jNs0VyRIQNmH-zB_wfMZ8syOgFdlzmF-u8gs3mCyQkZB9MW6JR1q3GhzkNHBHmLGo5l4IPTaTUnK06QaZFWvls3F-7ZLPE9dpaGAUFnbi2S4rIoAVR2nBkzQhYSP1vyU-nRCICbtQFK3mUJ5KQY5Q'
openai.api_key = hehezi
openai.api_base = "https://go-chatgpt-api.linweiyuan.com/imitate/v1"


def run(aspects, comments):
    result = {}
    for aspect in aspects:
        predicts_each_aspect = []
        for i, comment in enumerate(comments): 
            messages = [{
            "role": "user",
            "content": '请你按照要求完成文本分类任务。下面这段话中如果没提到`{}`这一方面, 请直接输出-2。如果提到了这一方面,请进一步分析这方面的情感倾向,如果是正面的请输出1,负面请输出-1,中性请输出0。 请特别注意,你必须且只需输出标准答案,即-2,-1,0,1四个数字之间的一个,不要输出其他额外的文字。\n'.format(aspect) + comment
            }]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                # stream=True
            )

            print(f'{i+1}_User: 第{i+1}条评论')
            print(f'{i+1}_Chatgpt: ',end='')
            
            for i, choice in enumerate(response.choices):
                ans = choice.message['content'].strip()
                try:
                    predicts_each_aspect.append(int(ans))
                    print(ans, end="", flush=True)
                except Exception:
                    print(f"choice{i} 输出了{ans}, 不是整数")
                    continue
                print('\n')
            time.sleep(1)
        result[aspect] = predicts_each_aspect
    
    return result


def test():
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
            "role": "user",
            "content": "你好"
            }
        ],
        stream=True
    )

    for chunk in response:
        print(chunk.choices[0].delta.get("content", ""), end="", flush=True)


def calculate_F1(a:list, b:list) -> float:
    f1 = f1_score(a, b, average='macro')
    return f1


if __name__ == '__main__':
    # test()
    aspects = ['交通是否便利','距离商圈远近','是否容易寻找','排队等候时间','服务人员态度','是否容易停车','点菜上菜速度','价格水平','性价比','折扣力度','装修情况','嘈杂情况','就餐空间','卫生情况','分量','口感','外观','推荐程度','本次消费感受','再次消费的意愿']
    df = pd.read_excel('./sample.xlsx')
    comments = df.iloc[:,2].to_list()
    y_s = df.iloc[:,3].to_dict()
    for i in y_s:
        y_s[i] = list(y_s[i].values())
    pred_s = run(aspects, comments)
    
    F1_total = 0
    for aspect_i, aspect_j in zip(y_s, pred_s):
        F1_each_aspect = calculate_F1(y_s[aspect_i],pred_s[aspect_j])
        F1_total += F1_each_aspect
        print(aspect_i, F1_each_aspect)
    
    F1_total /= 3
    print("F1_total: ",F1_total)
    