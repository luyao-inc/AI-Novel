# config_manager.py
# -*- coding: utf-8 -*-
import json
import os
import threading
from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter
import traceback
import time


def load_config(config_file: str) -> dict:
    """从指定的 config_file 加载配置，若不存在则返回空字典。"""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config_data: dict, config_file: str) -> bool:
    """将 config_data 保存到 config_file 中，返回 True/False 表示是否成功。"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        return True
    except:
        return False

def test_llm_config(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout, log_func, handle_exception_func):
    """测试当前的LLM配置是否可用"""
    def task():
        try:
            log_func("开始测试LLM配置...")
            print("开始测试LLM配置...")
            
            llm_adapter = create_llm_adapter(
                interface_format=interface_format,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

            log_func(f"API参数详情:")
            log_func(f"- 接口格式: {interface_format}")
            log_func(f"- 模型名称: {model_name}")
            log_func(f"- 基础URL: {base_url}")
            log_func(f"- 温度: {temperature}")
            log_func(f"- 最大Token: {max_tokens}")
            
            test_prompt = "请回复'你好，世界'"
            log_func(f"发送测试提示词: '{test_prompt}'")
            log_func("请等待API响应...")
            print(f"发送测试提示词: '{test_prompt}'")
            
            try:
                start_time = time.time()
                log_func("DeepSeekAdapter: 开始调用API...")
                response = llm_adapter.invoke(test_prompt)
                elapsed = time.time() - start_time
                
                # 确保GUI日志显示完整响应
                log_func(f"💬 API响应耗时: {elapsed:.2f}秒")
                
                if response:
                    log_func("✅ LLM配置测试成功！")
                    log_func(f"💬 API完整响应: \n{'-'*40}\n{response}\n{'-'*40}")
                    print(f"API完整响应: {response}")
                else:
                    log_func("❌ LLM配置测试失败：API返回为空")
                    print("❌ LLM配置测试失败：API返回为空")
            except Exception as api_err:
                log_func(f"❌ API调用失败: {str(api_err)}")
                log_func(f"错误详情: {traceback.format_exc()}")
                print(f"API调用失败: {str(api_err)}")
        except Exception as e:
            log_func(f"❌ LLM配置测试出错: {str(e)}")
            log_func(f"错误堆栈:\n{traceback.format_exc()}")
            handle_exception_func("测试LLM配置时出错", str(e), traceback.format_exc())

    threading.Thread(target=task, daemon=True).start()

def test_embedding_config(api_key, base_url, interface_format, model_name, log_func, handle_exception_func):
    """测试当前的Embedding配置是否可用"""
    def task():
        try:
            log_func("开始测试Embedding配置...")
            print("开始测试Embedding配置...")
            
            log_func(f"Embedding API参数详情:")
            log_func(f"- 接口格式: {interface_format}")
            log_func(f"- 模型名称: {model_name}")
            log_func(f"- 基础URL: {base_url}")
            
            try:
                from embedding_adapters import create_embedding_adapter  # 确保导入在执行时完成
                print(f"创建Embedding适配器: {interface_format}")
                log_func(f"创建Embedding适配器: {interface_format}")
                
                embedding_adapter = create_embedding_adapter(
                    interface_format=interface_format,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name
                )

                test_text = "测试文本"
                log_func(f"发送测试文本: '{test_text}'")
                log_func("请等待Embedding API响应...")
                print(f"发送Embedding测试文本: '{test_text}'")
                
                start_time = time.time()
                log_func("开始调用Embedding API...")
                embeddings = embedding_adapter.embed_query(test_text)
                elapsed = time.time() - start_time
                
                if embeddings and len(embeddings) > 0:
                    log_func("✅ Embedding配置测试成功！")
                    log_func(f"💬 请求耗时: {elapsed:.2f}秒")
                    log_func(f"💬 生成的向量维度: {len(embeddings)}")
                    log_func(f"💬 向量前5个值: \n{'-'*40}\n{embeddings[:5]}\n{'-'*40}")
                    print(f"向量长度: {len(embeddings)}, 前几个元素: {embeddings[:3]}")
                else:
                    log_func("❌ Embedding配置测试失败：返回的向量为空")
                    print("Embedding API返回为空")
            except Exception as api_err:
                log_func(f"❌ Embedding API调用失败: {str(api_err)}")
                log_func(f"错误详情: {traceback.format_exc()}")
                print(f"Embedding API调用失败: {str(api_err)}")
                raise
        except Exception as e:
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            log_func(f"❌ Embedding配置测试出错: {error_msg}")
            log_func(f"错误堆栈:\n{traceback_str}")
            handle_exception_func("测试Embedding配置时出错", error_msg, traceback_str)

    # 使用主线程安全的方式启动任务
    threading.Thread(target=task, daemon=True).start()