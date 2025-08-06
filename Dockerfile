# 基于python3.9 构建基础镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.9-slim

LABEL authors="chenm1xuexi"

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到容器的/app目录下
COPY . /app


# 创建虚拟环境
RUN python -m venv venv


# 激活虚拟环境并安装依赖
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量，使虚拟环境的bin目录位于PATH中
ENV PATH="/app/venv/bin:$PATH"

# 指定容器启动时运行的命令
CMD ["python", "chainlit_app.py"]
