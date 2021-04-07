# Learned knowledge

a diary recording the progress and learned knowledge.

[TOC]

- Python dictionary类型

- Python set()函数

  - 创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等

- Python 列表解析

- `if __name__ == "__main__":`的意思

- ``````
  if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
  ``````

- Json.load()与json.loads()的区别

- Python boolean类型，首字母大写

- os.walk，过滤隐藏文件

## 0328

- numpy.random.randint(low, high=None, size=None, dtype='l')
  - https://blog.csdn.net/u011851421/article/details/83544853
- python range()
- np.random.seed(100)
- Python 导入外部类
  - https://blog.csdn.net/laoyaotask/article/details/9164407
- heapq
  - nlargest
  - https://docs.python.org/zh-cn/3/library/heapq.html
- 匿名函数
  - 关键字 lambda
- from operator import itemgetter
  - https://blog.csdn.net/qq_22022063/article/details/79019294

## 0403

- git 更新远程分支到本地

  https://www.cnblogs.com/delav/p/11118555.html
  
- enumerate()函数

  https://www.runoob.com/python/python-func-enumerate.html


## 0404

- implicit feedback
  - https://jessesw.com/Rec-System/



## 0405

- Neighborhood models(user-based and item-based KNN)
- Latent factor models(with SVD)
- playlist_tracks.csv表，需对每个playlist的track去重
- 环境安装：
  - scikit-learn
  - scipy
- 必看
  - https://github.com/VasiliyRubtsov/recsys2018
  - https://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/KNNBasic_analysis.ipynb
  - https://blog.csdn.net/weixin_42608414/article/details/87712114?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-1&spm=1001.2101.3001.4242
  - https://blog.csdn.net/mouday/article/details/88181713?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-1&spm=1001.2101.3001.4242





## 0407

- remember: 在win系统上修改test_playlist_tracks.csv的column名为'tid'而不是'track_uri'

- 安装库implicit

  - ```
    conda install -c conda-forge implicit
    ```

- 安装ipywidgets
- 必读
  - https://blog.csdn.net/weixin_42608414/category_8685648.html?spm=1001.2014.3001.5482
- implicit
  - https://blog.csdn.net/weixin_42608414/article/details/90319447