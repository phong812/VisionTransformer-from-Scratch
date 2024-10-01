<h1> Build VisionTransformer to classify 50% of Tiny Food101 </h1>
<href> https://arxiv.org/abs/2010.11929 </href> <br>
<p1>Use: Pytorch, matplotlib, wandb, Flask ... </p1> <br>
Download dataset <br>
```
kaggle datasets download -d msarmi9/food101tiny 
``` <br>
Results: Compare to this paper with not lr warm-up, lr decay <br>

To run: with all hyperparameter are the same as the base version of model <br>

```
python main.py
```

<br>

![image](https://github.com/user-attachments/assets/7727df2f-97cb-48d6-8c88-0f23cace09db)



![image](https://github.com/user-attachments/assets/6ab1080d-25f9-4a00-81ce-46d47160671b)

Use pretrained model <br>

![image](https://github.com/user-attachments/assets/44c8b1f3-f963-44e1-a02a-4638513e011f)


![image](https://github.com/user-attachments/assets/5c410ff8-0aa5-4380-a372-1f4dad65ba58)

Use Flask to deploy <br>

![image](https://github.com/user-attachments/assets/0f909b48-5753-45b9-8a5f-26047bd7c605)








