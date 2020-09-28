# simple_bert_for_tf2
A bert layer for TF2.0 model.

## Itroduction
Bert is build as a TF.Keras Layer.  
  
Merits  
1.Easy to apply bert as a layer in a practical TF2.0 model.  
2.Using only numpy and Tensorflow2.0 as third party packages.  
  
Notes  
1.Comments are writen in Chinese.  
2.Dropout are removed according to the study of ALBERT.  
3.Transformer Block weights are shared cross layers according to ALBERT.  
4.Pretrain model loss is made up of MLM loss and the loss estimated based on the remaining words. According to the study of ELECTRA.  
5.No pretrained model language model provided here.  
6.Vocab.txt is squeezed for a certain project.  
  
  
  
由于想在实际任务中应用bert，而找到的bert tensorflow实现又让我踩了不少坑。  
因此就动手实现了一个稍微简洁清爽一些的TF2.0 bert。  
这里的bert继承了keras.layers.Layer类，实际应用时，可以方便地加到keras模型中。  
  
优点：  
1.写成了一个layer，无论是预训练还是finetune，用起来都方便;  
2.第三方库只用了TF2.0和numpy，你一定能跑起来;  
3.中文注释拉满，你一定能看懂每一步.  

注意：  
1.根据ALBER的研究，移除了dropout;  
2.根据ALBERT的研究，transformer层之间的参数共享（不共享效果好一点，共享后模型体积小，Inference time其实是一样的）;  
3.根据ELECTRA的研究，预训练时考虑没被mask的单词的损失可以加速模型效果提升（ELECTRA中是判断字符是否被替换），本项目中的loss也是由mlm loss和un-mask单词的loss两部分组成的。如觉不妥，可以直接在pretrain文件的loss定义里将后半部分的loss删去;  
4.没有提供用大语料加TPU烹饪的预训练模型，如果要读google bert预训练模型里的参数，应该也是可以的。
不过实际项目中，很可能要根据项目语料重新进行预训练的。  
5.vocab相较谷歌提供的原版，条目减少了很多。大多用不到，删了让token embedding的weights size减少了很多。可以根据实际项目情况，来替换vocab文件。  


## Files
```
|--bert_parts
|    |--layers.py       bert layer using keras TF2 | 基于keraslayer的bert layer, bert中的组件也被写成了layer放在此文件中
|    |--tokenizer.py    tokenizer for Chinese      | 用于对中文做tokenize的文件
|    |--vocab.txt       vocab file                 | 词典文件，用于将字符转换为token id
|--datasource.py        genarate data              | 产生数据
|--finetune.py          an example for finetune    | 微调的例子
|--pretrain.py          an example for pretrain    | 预训练的例子
```
  
  
## Instructions
```
Pretrain:
python pretrain.py

finetune:
python finetune.py

test bert_parts
python bert_parts/layers.py

test tokenizer(for Chinese text)
python bert_parts/tokenizer.py
```
