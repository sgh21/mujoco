import os
import test
try:
  import xml.etree.cElementTree as ET #尝试导入更快的cElementTree
except ImportError:
  import xml.etree.ElementTree as ET #失败则导入相对较慢的ElementTree
# import xml.etree.ElementTree as ET

tree = ET.parse('./project/students.xml')
root = tree.getroot()  # 使用getroot()获取根节点，得到的是一个Element对象

#root = ET.fromstring(country_data_as_string) #从字符串变量中读取，返回的是Element对象


########################## 访问XML ###################################
"""
tag = element.text      #访问Element标签
attrib = element.attrib #访问Element属性
text = element.text     #访问Element文本
"""

for element in root.findall('student'):
    tag = element.tag                  # 访问Element标签
    attrib = element.attrib            # 访问Element属性
    text = element.find('name').text   # 访问Element文本
    print(tag, attrib, text)

print(root[0][0].text)                 # 子节点是嵌套的，我们可以通关索引访问特定的子节点

########################## 查找元素 ###################################
print("============ Element.iter() ===============")
for student in root.iter('student'):   # Element.iter()
    print(student[0].text)


print("============ Element.findall() ============")
for element in root.findall('student'):# Element.findall()
    name = element.find('name').text
    age = element.find('age').text
    score = element.find('score').text

    print (name,age,score)

# tree = ET.parse("./project/test.xml")  # 将xml解析为树
# root = tree.getroot()       # 获取根节点
# for student in root.iter('student'):#会将节点中的子节点和孙节点都遍历一遍。
#   print (student[0].text) # 打印名字
#只会将节点中的子节点遍历一遍，不会查找孙节点。查找孙节点时可以使用下列方法：
# students = root.find('students')
# for element in students.findall('student'):
#   name = element.find('name').text
#   print name
num=5
for i in range(num):
    os.system('python test.py')