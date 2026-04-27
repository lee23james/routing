import argparse
import json
from fastapi import FastAPI, Request
import torch
import uvicorn
import datetime
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["TRITON_MMA"] = "0"

app = FastAPI()

# Note: Use CUDA_VISIBLE_DEVICES environment variable to specify GPUs
# Example: CUDA_VISIBLE_DEVICES=6,7 python server_vllm.py ...

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        max_length = json_post_raw.get('max_tokens', 512)
        top_p = json_post_raw.get('top_p', 0.95)
        temperature = json_post_raw.get('temperature', 0.6)
        messages = json_post_raw.get('messages')
        repetition_penalty = json_post_raw.get('repetition_penalty', 1.0)
        think_mode = json_post_raw.get('think_mode', False)
        stop = json_post_raw.get('stop', None)
        
        # logprobs 支持
        logprobs_enabled = json_post_raw.get('logprobs', False)
        top_logprobs = json_post_raw.get('top_logprobs', 0)
        
        # extra_body 参数
        extra_body = json_post_raw.get('extra_body', {})
        add_generation_prompt = extra_body.get('add_generation_prompt', True)
        continue_final_message = extra_body.get('continue_final_message', False)

        # 构建 SamplingParams
        sampling_kwargs = {
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'max_tokens': max_length,
        }
        
        if stop is not None:
            sampling_kwargs['stop'] = stop
        
        if logprobs_enabled and top_logprobs > 0:
            sampling_kwargs['logprobs'] = top_logprobs
        
        sampling_params = SamplingParams(**sampling_kwargs)

        # Qwen3 系列模型的 think 模式支持
        qwen3_models = [
            '/export/yuguo/ppyg2/model/qwen3-8b',
            '/export/yuguo/ppyg2/model/qwen3-4b',
            '/export/yuguo/ppyg2/model/qwen3-1.7b',
            '/export/yuguo/ppyg2/model/qwen3-0.6b',
            '/export/yuguo/ppyg2/model/qwen3-14b',
            '/export/yuguo/ppyg2/model/qwen3-32b',
        ]
        # DeepSeek-R1 系列模型
        deepseek_r1_models = [
            '/export/yuguo/ppyg2/model/DeepSeek-R1-1.5B',
            '/export/yuguo/ppyg2/model/DeepSeek-R1-14B',
        ]
        if args.model in qwen3_models:
            if think_mode:
                tokenizer.chat_template = chat_template["Qwen3_8b_think_chat_template"]
            else:
                tokenizer.chat_template = chat_template["Qwen3_8b_nothink_chat_template"]
        elif args.model in deepseek_r1_models:
            if think_mode:
                tokenizer.chat_template = chat_template["DeepSeek_R1_think_chat_template"]
            else:
                tokenizer.chat_template = chat_template["DeepSeek_R1_nothink_chat_template"]

        # 应用 chat template
        # DeepSeek-R1 + continue_final_message 时手动拼接，
        # 因为其 chat template 会删除 <think> 内容导致 continue_final_message 匹配失败
        if args.model in deepseek_r1_models and continue_final_message:
            # 手动拼接 DeepSeek-R1 格式的 prompt
            system_prompt = ''
            parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                elif msg['role'] == 'user':
                    parts.append('<｜User｜>' + msg['content'])
                elif msg['role'] == 'assistant':
                    # 续写模式：直接拼接 content，不加 <｜end▁of▁sentence｜>
                    parts.append('<｜Assistant｜>' + (msg.get('content') or ''))
            inputs = tokenizer.bos_token + system_prompt + ''.join(parts)
        else:
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message
            )
        print(f"inputs: {inputs}")
        outputs = model.generate(inputs, sampling_params)
        
        output = outputs[0].outputs[0]
        token_ids = output.token_ids
        token_count = len(token_ids)
        response_text = output.text

        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建 OpenAI 兼容的响应格式
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": output.finish_reason if hasattr(output, 'finish_reason') else "stop"
        }
        
        # 如果启用了 logprobs，添加 logprobs 信息
        if logprobs_enabled and output.logprobs is not None:
            logprobs_content = []
            for idx, (token_id, logprob_dict) in enumerate(zip(token_ids, output.logprobs)):
                token_str = tokenizer.decode([token_id])
                
                # 获取 top logprobs
                top_logprobs_list = []
                if logprob_dict is not None:
                    sorted_logprobs = sorted(logprob_dict.items(), key=lambda x: x[1].logprob, reverse=True)[:top_logprobs]
                    for tok_id, logprob_info in sorted_logprobs:
                        top_logprobs_list.append({
                            "token": tokenizer.decode([tok_id]),
                            "logprob": logprob_info.logprob,
                            "bytes": None
                        })
                
                logprobs_content.append({
                    "token": token_str,
                    "logprob": logprob_dict[token_id].logprob if logprob_dict and token_id in logprob_dict else 0.0,
                    "bytes": None,
                    "top_logprobs": top_logprobs_list
                })
            
            choice["logprobs"] = {"content": logprobs_content}
        else:
            choice["logprobs"] = None
        
        answer = {
            "id": f"chatcmpl-{now.timestamp()}",
            "object": "chat.completion",
            "created": int(now.timestamp()),
            "model": args.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(inputs)),
                "completion_tokens": token_count,
                "total_tokens": len(tokenizer.encode(inputs)) + token_count
            }
        }
        
        log = f"[{time_str}] think mode: {think_mode}, logprobs: {logprobs_enabled}, prompt: {messages[:100]}..., response: {repr(response_text[:100])}..."
        print(log)
        return answer

    except Exception as e:
        import traceback
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return {"error": error_message}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom port and model")
    parser.add_argument('--model', type=str, required=True, help="Model to load (e.g., 'microsoft/Phi-3-mini-4k-instruct')")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    parser.add_argument('--tensor-parallel-size', type=int, default=None, help="Number of GPUs to use for tensor parallelism (default: use all available GPUs)")
    return parser.parse_args()

if __name__ == '__main__':
    # Command line arguments
    args = parse_args()

    # Model and tokenizer loading based on provided model name
    model_dir = args.model

    with open("chat_template.json", "r", encoding="utf-8") as f:
        chat_template = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # Determine tensor parallel size
    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else torch.cuda.device_count()
    print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    
    model = LLM(model = args.model, 
                tensor_parallel_size=tensor_parallel_size, 
                trust_remote_code=True,
                max_model_len=16384,  # 减少最大序列长度以节省KV cache内存
                gpu_memory_utilization=0.90,  # 限制GPU内存使用率
                enforce_eager=True,  # 禁用CUDA Graph以节省内存
                )

    # Running the server with the specified port
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)