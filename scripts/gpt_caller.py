import json
import requests
from functools import cache
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sseclient
import time

# openai_api = os.environ.get('OPENAI_API_ADDR') if os.environ.get('OPENAI_API_ADDR') else "https://api.openai.com"
openai_api = 'http://49.51.186.136'
# openai_api = 'http://49.51.186.136:82'
print("OPENAI_API_ADDR:" + openai_api)


class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code


class Gpt4Caller():
    def __init__(self, api_key=None):
        retry_strategy = Retry(
            total=5,  # 最大重试次数（包括首次请求）
            backoff_factor=1,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
            allowed_methods=["POST"]  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key

    def post_url(self, params_gpt, type='chat', headers={}):
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        if type == 'chat':
            url = openai_api + '/v1/chat/completions'
        elif type == 'embedding':
            url = openai_api + '/v1/embeddings'
        response = self.session.post(url, stream=params_gpt.get('stream', False), headers=headers, data=json.dumps(params_gpt))
        return response

    def get_response(self, prompt, sys_prompt="", model_name='gpt-4-1106-preview', history=[]):
        params_gpt = {
            "model": model_name,
            "messages": [{"role": "user", "content": ''}],
        }
        params_gpt['messages'][0]['content'] = prompt
        if sys_prompt != "":
            params_gpt['messages'].insert(0, {"role": "system", "content":sys_prompt})
        if history:
            for msg in history:
                params_gpt['messages'].insert(-1, msg)
        response = self.post_url(params_gpt)
        first_token = None
        if response.status_code != 200:
            err_msg = "access openai error, status code: %s" % (response.status_code)
            raise OpenAIApiException(err_msg, response.status_code)
        data = json.loads(response.text)['choices'][0]['message']['content']
        return data

    def get_response_stream(self, prompt, sys_prompt="", model_name='gpt-4-1106-preview', history=[]):
        params_gpt = {
            "model": model_name,
            "messages": [{"role": "user", "content": ''}],
            "stream": True 
        }
        params_gpt['messages'][0]['content'] = prompt
        if sys_prompt != "":
            params_gpt['messages'].insert(0, {"role": "system", "content":sys_prompt})
        if history:
            for msg in history:
                params_gpt['messages'].insert(-1, msg)
        response = self.post_url(params_gpt)
        first_token = None
        client = sseclient.SSEClient(response)
        for event in client.events():
            if not first_token:
                first_token = time.time()
            if event.data != '[DONE]':
                event_json_data = json.loads(event.data)
                finish_reason = event_json_data['choices'][0]['finish_reason']
                if not finish_reason:
                    data = event_json_data['choices'][0]['delta']['content']
                    yield data

    @cache
    def get_embedding(self, input, model_name='text-embedding-ada-002', headers={}):
        params_gpt = {
            "model": model_name,
            "input": input
        }
        response = self.post_url(params_gpt, type='embedding')
        data = json.loads(response.text)
        return data

if __name__ == '__main__':
    proxy = Gpt4Caller()
    prompt = 'tell me a story'
    for data in proxy.get_response_stream(prompt):
        print(data)