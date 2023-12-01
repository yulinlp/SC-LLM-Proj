关联 https://github.com/yulinlp/SC-Mini-Proj ，是Task2和3的部分实现  
默认只处理 `交通是否便利` 这一方面的数据

关于chatgpt伪API式访问方法, 参考 https://github.com/linweiyuan/go-chatgpt-api

### chatgpt_v1
- zero-shot 
- F1 = 0.2236   
### chatgpt_v2
- CoT_v1 + 1-shot
- F1 = 0.3313
### chatgpt_v3 (还在调试, 暂不公开代码)
- Cot_v2 + 2-shot
- F1 = 0.4659
