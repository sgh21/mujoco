# main
## 环境和版本
+ 安装环境：Ubuntu 20.04.06
+ 编辑器：VScode
+ 软件版本：mujoco210,mujoco_py(2.1.2.14),Anaconda(2020.02_linux_x86)
## 软件简介
+ mujoco210：仿真引擎
+ mujoco_py:mujoco的python接口由openAI开发，也可直接用mujoco3.1.3(google开发的python接口)
+ Anaconda：虚拟环境控制，将mujoco安装环境和主环境分离，便于包管理
## 一些参考
+ mujoco3.1.3安装：pip install mujoco
+ mujoco,mujoco_py安装:https://blog.csdn.net/weixin_51844581/article/details/128454472
+ anaconda安装：https://cloud.tencent.com/developer/article/1649008
## 一些bug
+ cython.compiler.errors.compileerror:默认的cython版本有一点问题，卸载重装
```cpp
pip uninstall cython
pip install cython==0.29.21
```
+ No such file or directory: ‘patchelf’:没有安装patchelf(某动态库链接库修改器),aot安装即可
```cpp
sudo apt-get install patchelf
```
+ Missing GL version：找不到GL模块，需要将拓展添加到.bashrc后重新source
```cpp
sudo gedit ~/.bashrc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
source ~/.bashrc
```
## 运行example
+ 安装相应依赖(pip install <package name>)
    + simple_pid: 提供了一个简单但功能强大的PID控制器实现
    + termcolor: 用于在终端中添加颜色和样式,想必是学姐用于美化终端的吧
    + pyquaternion: 处理四元数的Python库
    + cv2: opencv 图像处理
    + matplotlib: Python 数据可视化库，用于创建出版质量的图形
    + transforms3d:三维坐标转换库
    + pybullet:项目中使用pybullet提供的运动学逆解器
+ 如果仍有其他依赖需要安装，按照错误提示进行安装即可
+ 运行example.py：
    + 需要进入my_mujoco_platform文件夹中
    + 正常运行python文件即可，会自动输出一些图片数据