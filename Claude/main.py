from Claude.claude import Claude
import pandas as pd
import sys

log = open('../chat.log', "a", encoding='utf-8')
sys.stdout = log
ai = Claude()

df = pd.read_excel('./sample.xlsx')
comments = df.loc[:,'content'].to_list()

aspects = ['交通是否便利','距离商圈远近','是否容易寻找','排队等候时间','服务人员态度','是否容易停车','点菜上菜速度','价格水平','性价比','折扣力度','装修情况','嘈杂情况','就餐空间','卫生情况','分量','口感','外观','推荐程度','本次消费感受','再次消费的意愿']
for aspect in aspects:
    prompt = "下面我会给你一些食客的评论。请先分析每段话中是否提到了`{}`这一方面,如果没提到请直接输出-2。如果提到了这一方面,请进一步分析食客对于这方面的情感倾向,如果是正面的请输出1,负面请输出-1,中性请输出0。 请特别注意,你必须且只需输出标准答案,即-2,-1,0,1四个数字之间的一个,不要输出其他额外的文字。如果你听懂了, 请回复我听懂了".format(aspect)
    response = ai.talk_once(prompt)
    
    print("User: \n",prompt)
    print("Assistant: \n",response)
    print("="*30)
    
    res_each_aspect = []
    for idx, comment in enumerate(comments):
        response = ai.talk_once(comment)
        print("User{}: \n".format(idx),comment)
        print("Assistant{}: \n".format(idx),response)
        try:
            res_each_aspect.append(int(response))
        except Exception:
            print("WARNING!!!\nClaude输出不符合规则")
        
    print(res_each_aspect)

    
