from datetime import datetime

import openai
import pandas as pd
import sys
from sklearn.metrics import f1_score
import time

# log = open('chat.log', "a", encoding='utf-8')
# sys.stdout = log

aspects = ['交通是否便利', '距离商圈远近', '是否容易寻找', '排队等候时间', '服务人员态度', '是否容易停车',
           '点菜上菜速度', '价格水平', '性价比', '折扣力度', '装修情况', '嘈杂情况', '就餐空间', '卫生情况', '分量',
           '口感', '外观', '推荐程度', '本次消费感受', '再次消费的意愿']
zouzou = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJ6eXgucmlja3l5QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLXprZHdDaGVrdEN4ZFhMOHV3bXRnTmJYTSJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjQ2ZjEzMjQzMzUxNDJhMjcxMDZjNzgxIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5NTYyNjk5OSwiZXhwIjoxNjk2ODM2NTk5LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIGVtYWlsIHByb2ZpbGUgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.VSwiMuZRdB8xv8cBfyaSe26FDxbtmpw9vB2l-N1zJYzpOPGn0a_6-se3Y93Gt9f_d-1Lnpg-PEK-XUy7Pakt0cL0o8uHr2h1TV4qXrTEK4hsJRRLJs1vV1CKatM8CWm9gvOKSa_6aVkcC_MPbyhMnqYxo5jHq8PFFfrD5Iyu_SYB6VK1QtjFU19gXA4GHkFtNWLSGPTuFDaDKMayJ9fo4EcGkiL-SQL4_XiyVzFBQebzMqvITkK_iYAQ0eXFkaWancFa_WNXHjK6aWa1f_Y3NFC_ig0NJRAsFF5xgjc0Yl61DT4bBaH5HjNH61XI-a4FaCJAtb6GWBraZirh6exejg'
lin = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiIxNTY1OTE4NDEyQHFxLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctS2NrNDRSSXU0MWt4amd3NTBHM2FTRTRoIiwidXNlcl9pZCI6InVzZXIta1hlQ1lFR0R3Y2R1RHVNYkVYZng4Mjl0In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2MzliMjRkNTk3OTk2ZDlmZGZiMWRiYmMiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNjk3ODE1MTE5LCJleHAiOjE2OTg2NzkxMTksImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.CXOFEdAuNkJirNHqd4x4c4WHUauSkxMhTQAPxBzf1_Vr8mbqirx9yMP0QiOuf7RO91VxaTObGmh1D4qTB2deSD1vTA8ROHvy6X_RCsl3AOBFgFufW_FT9q8_RYvA53JFimEdQkAsrzKEQG4e9x5bsTI6MzHe-z1hOsNHywhFUtU6Mp-iyKfxsJk9JCWWMJRdnSkPeGeO8LhTxjz3wNhWbO83SAO93oVcUV53n3kxlCOm_vTAaSHtHXtYBBZZEQoJ2wiDSa4XqjTsgCEs9nShNmpWuvSamkzK_SfCmOABbOqQaK-CGEm3fNHc9gDDHHSX79_vnWgEEq9NoI-gcYbvmg'
hehezi = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtdXl1ZW5ob3NoaW5vQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctMTJzdTJsWWExb3pJdHF3M2ViQ3hSUlhHIiwidXNlcl9pZCI6InVzZXItcTc2eU03RjExZTVYdnBOSGRMcTAxRFpqIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMjU4MDI5OTU1OTUzMzYyNjAxMSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2OTc1NDQ1ODgsImV4cCI6MTY5ODQwODU4OCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBlbWFpbCBwcm9maWxlIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.yFS-6KR97X8Do3cyHu0ibUGsQAI39BWrT0EQP2ngGtqST2o7dP-_N_5gi8bKE7K3HUWpiuLKQcF7iJ9mqVD9NUbdEDnjMAB6PVzlLyoSIDT2-HnDxV8_ZHn1gIYtmapPGu-AWd1hrQm2Mxi86_WXEowvY1dNcnTQvBCEl3XajsIlrxYZqpjy97b4Hb-mJJo99OoscD3EU10KEBV58Ti86VWgmlOB1oQPWb5LMQpyVv62ntBCyrd8IsBjdwi-uUQ3xpbn124lulBWMFvebtpryZGll26BwHZ0GXJX8q5XGjiXXJLtKEV4I22PAvUSdRbV1pIP6H6F00nOiO4NRzUCiA'
openai.api_key = hehezi
openai.api_base = "https://go-chatgpt-api.linweiyuan.com/imitate/v1"


def calculate_F1(a: list, b: list) -> float:
    f1 = f1_score(a, b, average='macro')
    return f1


class Chatgpt:
    def __init__(self, aspect_idx=0):
        df = pd.read_excel('./data/sample.xlsx', nrows=30)
        self.aspect = aspects[aspect_idx]
        self.comments = df.iloc[:, 2].to_list()
        self.y_s = df[self.aspect].to_list()

    def predict(self):
        for i, comment in enumerate(self.comments):
            time.sleep(6)
            prompt = f'''现在你是一个文本分类大师，请按照下面的要求完成文本分类任务。
下面这段话中如果没提到`{self.aspect}`这一方面, 请直接输出-2。 如果提到了这一方面,请进一步分析这方面的情感倾向。如果是正面的请输出1,负面请输出-1,中性请输出0。 
下面是一个例子： 
问： "这几天有天蝎座打折活动 1-2个天蝎打6.9折 不包括锅底 蘸料和饮料 于是就带着闺蜜跑来大吃大喝 首先地点就在地铁站旁边的百联世茂 非常好找 门面很漂亮 
今天人超多啊 排了1个多小时 有点遗憾没吃到推荐的吊龙 #锅底#我们分别要了#今牛锅#和#牛腩锅# 差别就是前一个有一个牛丸和一块牛腩 后一个牛腩多一点没有牛丸 锅底的汤很好喝 
里面有枸杞和白萝卜 #手工牛筋丸#强烈推荐 超Q 非常筋道 吃着感觉超赞 和工业做出来的一吃就不一样 赞赞赞 #嫩肉#因为都没有吊龙 匙柄什么的 所以就要了这个 不过味道还是很赞的 
肉很新鲜 肉质很紧 #牛腩#排队超过一小时就送 肉切的比较大块一点 但是超容易熟 不过感觉稍微有点干 #炸豆皮#据广东的小伙伴说这是他们吃火锅必点的 第一次感觉还不错 有点脆脆的 
不过不能吃太多 不然会有点腻 #蔬菜拼盘#比想象中的要少 有菠菜 白菜 还有个什么菜 我的惯性思维告诉我不是应该有金针菇什么的吗 #蘸料#东西并不多 不过自制辣酱挺好吃的 
还有沙茶酱什么的 #杨梅汁#一扎 味道马马虎虎 比较清爽" 
答: "原文中说`地点就在地铁站旁边的百联世茂`，所以这段话中提到了`交通是否便利`这一方面，所以需要进一步分析`交通是否便利`这方面的情感倾向。由于原文说`在地铁站旁边`，推断出交通很便利，情感倾向是正面的，因此最后的结果是1"

请完成下面的任务，你不需要重复问题，只需要回答：
问：{comment}
答：'''
            messages = [{
                "role": "user",
                "content": prompt
            }]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                # stream=True
            )

            ans = response.choices[0].message['content']
            print(f'{i + 1}_User: 第{i + 1}条评论')
            print(f'{i + 1}_Chatgpt: {ans}')


if __name__ == '__main__':

    # current_datetime = datetime.now()
    # current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # print(current_datetime_str, "=" * 30)
    #
    bot = Chatgpt()
    # bot.predict()
    ans = [-2, -2, 1, 1, -2, 0, -1, -2, 1, -2, -2, -1, 1, -2, -2, 0, 1, -2, -2, 1, -2, -2, 1, -2, 1, -2, 1, 1, 1, -2]
    F1 = calculate_F1(ans, bot.y_s)
    print(F1)