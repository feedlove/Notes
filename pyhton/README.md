我们在使用Python时经常需要安装各种模块，而pip是一个很强大的模块安装工具，类似于linux的Yum一样，安装模块时能自动解决依赖等，总结如下：

 

一、安装pip：

　　安装python时自动会安装pip

 

二、使用pip：

　　1、升级pip

　　　　python -m pip install --upgrade pip

　　2、安装模块

　　　　pip install gevent　　#安装指定模块

　　　　pip install -r  requirements.txt     # 安装requirements.txt文件中定义的模块列表

 

三、更改pip源

　　pip安装第三方模块时，默认从Python官方模块库：https://pypi.python.org下载，但由于经常被防火墙所挡或下载速度慢等原因，因此需要更换下国内镜像

 

　　网上有很多可用的源，例如

　　　　豆瓣：http://pypi.douban.com/simple/
　　　　清华：https://pypi.tuna.tsinghua.edu.cn/simple

　　最近使用得比较多并且比较顺手的是清华大学的pip源，它是官网pypi的镜像，每隔5分钟同步一次，地址为 https://pypi.tuna.tsinghua.edu.cn/simple

　　1、临时使用：

　　　　可以在使用pip的时候加参数-i https://pypi.tuna.tsinghua.edu.cn/simple

　　　　例如：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gevent，这样就会从清华这边的镜像去安装gevent库。

　　2、永久修改，一劳永逸：


　　　　Linux下，修改 ~/.pip/pip.conf (没有就创建一个)， 修改 index-url至tuna，内容如下：


　　　　[global]
　　　　index-url = https://pypi.tuna.tsinghua.edu.cn/simple

　　　　Windows下，直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini，内容如下

　　　　[global]
　　　　index-url = https://pypi.tuna.tsinghua.edu.cn/simple