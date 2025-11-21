from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from typing import List, Tuple, Optional, Dict, Union
from openai import OpenAI
import json
import copy
from pydantic import BaseModel, Field
import re
import os
import langProBe.constants as constants
import logging
from .synced_mcp_client import SyncedMcpClient

TOOL_PROMPT = """
## Tool Calling Rules
When external tools are required, the call request must be strictly generated according to the following rules:
<tool>  
{  
  "server_name": "",  
  "tool_name": "",  
  "inputs": {  
    "<parameter1>": "<value1>",  
    "<parameter2>": "<value2>",  
  }  
}  
</tool>  

If no tool is called, provide the final answer directly.

"""
            
class ProcessManager(BaseModel):
    id: str = Field(
        default=None,
        description="The ID of the process.",
    )
    lm_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API Key"
    )
    lm_api_base: str = Field(
        default=os.getenv("OPENAI_API_BASE"),
        description="OpenAI API Base URL"
    )
    model: str = Field(
        default=None,
        description="OpenAI Model Name, with prefix 'openai/'"
    )
    lm_usages: List[Dict] = Field(
        default=[],
        description="Usage statistics for the model"
    )
    mcp_rts: List[Dict] = Field(
        default=[],
        description="Usage statistics for the MCPs"
    )
    mcp_retry_times: List[Dict] = Field(
        default=[],
        description="Statistics for the MCP retries"
    )


class MCPCall(BaseModel):
    mcp_server_name: Optional[str] = None
    mcp_tool_name: Optional[str] = None
    mcp_args: Optional[Dict] = None


class MCPCallList(BaseModel):
    shutdown: bool = False
    mcps: Optional[List[MCPCall]] = None
    raw_content: Optional[str] = None

@retry(
    stop=stop_after_attempt(5),  
    wait=wait_exponential(multiplier=1, min=2, max=10),  
    reraise=True,
)
def call_lm(
            messages: List, 
            manager: ProcessManager, 
            logger: logging.Logger, 
            temperature: float|None=None,
            ) -> tuple[str | None, int, int]:    
    
    try:
        oai = OpenAI(
            api_key=manager.lm_api_key,
            base_url=manager.lm_api_base,
        )
        prefix, model_name = manager.model.split('/')
        assert prefix == 'openai'

        if model_name in ['deepseek-r1', 'qwq-plus', 'qwq-32b']: # qwen reasoning模型仅支持流式输出
            reasoning_content = ""  # 定义完整思考过程
            answer_content = ""     # 定义完整回复
            is_answering = False   # 判断是否结束思考过程并开始回复

            completion = oai.chat.completions.create(
                model=model_name, 
                messages=messages,
                stream=True,
                stream_options={
                    "include_usage": True
                }
            )
            for chunk in completion:
                # 如果chunk.choices为空，则打印usage
                if not chunk.choices:
                    usage = chunk.usage
                else:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                        answer_content += delta.content
            completion_tokens = usage.completion_tokens
            prompt_tokens = usage.prompt_tokens
            manager.lm_usages.append({
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
            })
            return '<think>' + reasoning_content + '</think>' + answer_content, completion_tokens, prompt_tokens


        if temperature is not None:
            response = oai.beta.chat.completions.parse(
                messages=messages,
                model=model_name,
                temperature = temperature
            )
        else:
            response = oai.beta.chat.completions.parse(
                messages=messages,
                model=model_name,
            )
            # print("Response is " + str(response))
        response_text = response.choices[0].message.content
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        manager.lm_usages.append({
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
            })
        return response_text, completion_tokens, prompt_tokens
    
    except Exception as e:
        logger.error(f"ID: {manager.id}, Error in call_lm: {str(e)}")
        if response:
            logger.error(f"ID: {manager.id}, Response: {response}")
        raise

def build_system_content(base_system: str,
                        mcps: List,
                        ) -> str:
    tools_section = "## Available Tools\n"
    for mcp in mcps:
        tools_section += f"### Server '{mcp['name']}' include following tools\n"
        if mcp['name'] in ['wuying-agentbay-mcp-server', 'Playwright']:
            tools_section += f"当使用本server来执行搜索任务时，请以https://www.baidu.com为初始网站进行搜索。"
        url = mcp.get("url")
        headers = mcp.get("headers", {})
        if not url:
            try:
                port = mcp.get('run_config')[0]["port"]
                url = f"http://localhost:{port}/sse"
            except:
                raise Exception("No url found")
        client = SyncedMcpClient(server_url=url, headers=headers)
        try:
            result = client.list_tools()
            tools = result.tools
        except Exception as e:
            raise Exception(f"Fail access to server: {mcp['name']}, error: {e}")

        for t in tools:
            tools_section += f"- {t.name}: {t.description}\n"
            input_schema = t.inputSchema
            required_params = input_schema.get("required", [])
            params_desc = []

            if "properties" in input_schema:
                for param_name, param_info in input_schema["properties"].items():
                    is_required = param_name in required_params
                    param_type = param_info.get("type", "")
                    param_desc = param_info.get("description", "")

                    req_tag = "必填" if is_required else "可选"
                    params_desc.append(
                        f"- {param_name} ({param_type}, {req_tag}): {param_desc}"
                    )

            # 使用更丰富的描述
            params_text = "\n".join(params_desc) if params_desc else "无参数"
            tools_section += f"  参数:\n{params_text}\n\n"

    prompt = base_system + f"""{tools_section}""" + TOOL_PROMPT

    return prompt


def build_init_messages(
        base_system: str,
        mcps: List,
        user_question: str,
       ) -> List[Dict]:
    system_content = build_system_content(base_system, mcps)
    messages = [
        {
            constants.ROLE: constants.SYSTEM,
            constants.CONTENT: system_content
        },
        {
            constants.ROLE: constants.USER,
            constants.CONTENT: user_question
        }
    ]
    return messages



def build_messages(
        messages: List[Dict],
        message_to_append: List[Dict],
        ) -> List[Dict]:
    assert messages[0][constants.ROLE] == constants.SYSTEM
    
    final_message = copy.deepcopy(messages)

    if message_to_append:
        if message_to_append[-1][constants.ROLE] == constants.USER:
            assert len(message_to_append) == 1
            assert final_message[-1][constants.ROLE] in {constants.ASSISTANT, constants.TOOL, constants.SYSTEM}
            final_message.extend(message_to_append)
        elif message_to_append[-1][constants.ROLE] == constants.ASSISTANT:
            assert len(message_to_append) == 1
            assert final_message[-1][constants.ROLE] in {constants.USER, constants.TOOL}
            final_message.extend(message_to_append)
        elif message_to_append[-1][constants.ROLE] == constants.TOOL:
            assert len(message_to_append) == 2
            assert final_message[-1][constants.ROLE] in {constants.USER, constants.TOOL}
            final_message.extend(message_to_append)
    
    # TODO: 超过最长上下文长度处理

    return final_message



def response_parsing(content: str) -> MCPCallList:
    pattern = r'<tool>(.*?)<\/tool>'
    matches = re.findall(pattern, content, re.DOTALL)
    mcps = []
    for match in matches:
        # TODO: 错误处理
        data = json.loads(match)
        mcps.append(MCPCall(
            mcp_server_name=data['server_name'].strip(),
            mcp_tool_name=data['tool_name'].strip(),
            mcp_args=data['inputs']
        ))

    if mcps:
        return MCPCallList(shutdown=False, mcps=mcps, raw_content=content)
    else:
        return MCPCallList(shutdown=True, mcps=None, raw_content=content)


def mcp_calling(
        mcp_call_list: MCPCallList,
        manager: ProcessManager,
        logger: logging.Logger,
) -> List[Dict]:
    logger.debug(f"ID:{manager.id}, Entering mcp_calling with mcp_call_list: {mcp_call_list}")

    if mcp_call_list.shutdown:
        logger.info(f"ID:{manager.id}, Shutdown flag is set. No more MCP calling.")
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else '',
            }
        ]
        logger.debug(f"ID:{manager.id}, Shutdown messages prepared: {messages}")
        return messages
    else:
        logger.info(f"ID:{manager.id}, Processing MCP call list with {len(mcp_call_list.mcps)} MCPs.")
        mcp_list = mcp_call_list.mcps
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else '',
                constants.TOOL_CALLS: []
            }
        ]
        result_str = ""
        for idx, mcp in enumerate(mcp_list, start=1):
            logger.debug(f"ID:{manager.id}, Processing MCP #{idx}: {mcp}")
            mcp_server_name = mcp.mcp_server_name
            mcp_tool_name = mcp.mcp_tool_name
            mcp_args = mcp.mcp_args

            tool_call = {
                "type": "function",
                "function": {
                    "name": mcp_tool_name,
                    "arguments": json.dumps(mcp_args, ensure_ascii=False)
                }
            }
            messages[0][constants.TOOL_CALLS].append(tool_call)
            logger.info(f"ID:{manager.id}, Calling MCP Server: {mcp_server_name}, Tool: {mcp_tool_name}, Arguments: {mcp_args}")

            # Manage manager.mcp_rts and manager.mcp_retry_times
            from langProBe.evaluation import global_config
            try:
                parsed_data = global_config

                target_name = mcp_server_name
                port = None
                url = None
                headers = {}  # Initialize headers before the loop
                for item in parsed_data.get("mcp_pool", []):
                    if item.get("name") != target_name:
                        continue

                    # Get headers from the item (should be done before break)
                    headers = item.get("headers", {})
                    
                    url = item.get("url", "")
                    if url:
                        logger.debug(f"ID:{manager.id}, Found URL for MCP Server '{target_name}': {url}")
                        break
                    
                    run_configs = item.get("run_config", [])
                    for config in run_configs:
                        port = config.get("port")
                        if port:
                            url = f"http://localhost:{port}/sse"
                            logger.debug(f"ID:{manager.id}, Constructed URL for MCP Server '{target_name}': {url}")
                            break
                    if url:
                        break

                if not url:
                    logger.error(f"ID:{manager.id}, No valid URL found for MCP Server '{target_name}'.")
                    raise ValueError(f"ID:{manager.id}, No valid URL found for MCP Server '{target_name}'.")

                client = SyncedMcpClient(server_url=url, headers=headers)
                logger.debug(f"ID:{manager.id}, Initialized SyncedMcpClient with URL: {url}")
                client.list_tools()
                logger.debug(f"ID:{manager.id}, Retrieved tool list from MCP Server '{target_name}'.")
            except Exception as e:
                logger.error(f"ID:{manager.id}, Failed to initialize SyncedMcpClient for server '{mcp_server_name}': {str(e)}")
                client = None

            if client:
                try:
                    logger.debug(f"ID:{manager.id}, Calling tool '{mcp_tool_name}' with arguments: {mcp_args}")
                    result = client.call_tool(mcp_tool_name, mcp_args)
                    texts = [item.text for item in result.content]
                    result_str_segment = ''.join(texts)
                    logger.debug(f"ID:{manager.id}, Received result from tool '{mcp_tool_name}': {result_str_segment}")

                    logger.info(f"ID:{manager.id}, MCP Server '{mcp_server_name}' returned: {result_str_segment[:5000]}")

                    result_str += result_str_segment
                except Exception as e:
                    logger.error(f"ID:{manager.id}, Error calling tool '{mcp_tool_name}' on MCP Server '{mcp_server_name}': {str(e)}")
            else:
                logger.warning(f"ID:{manager.id}, Skipping tool call for '{mcp_tool_name}' due to client initialization failure.")

        messages.append({
            constants.ROLE: constants.TOOL,
            constants.CONTENT: result_str[:5000],
        })
        logger.debug(f"ID:{manager.id}, Final messages prepared: {messages}")
        logger.info(f"ID:{manager.id}, mcp_calling completed successfully.")
        return messages

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )
