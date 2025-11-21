#!/bin/bash
 # 检查是否提供了配置文件路径参数
 if [ -z "$1" ]; then
   echo "Usage: $0 <config_file_path>"
   exit 1
 fi

 # 构造完整路径
 CONFIG_FILE="$1"
 if [[ ! "$CONFIG_FILE" == /* ]]; then
   CONFIG_FILE="configs/$CONFIG_FILE"
 fi



# 使用更直接的方法启动评估程序，确保多进程正确初始化
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache \
python -c "
import multiprocessing as mp
mp.set_start_method('spawn', True)
from langProBe.evaluation import main
main()
" \
--benchmark=WebSearch \
--dataset_mode=test \
--dataset_path=langProBe/WebSearch/data/websearch_600.jsonl \
--file_path=evaluation_websearch_test \
--lm=openai/deepseek-v3 \
--lm_api_base=https://dashscope.aliyuncs.com/compatible-mode/v1 \
--lm_api_key=xx \
--missing_mode_file=logs/task_messages.jsonl \
--num_threads=1 \
--config=$CONFIG_FILE