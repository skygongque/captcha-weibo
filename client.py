import requests
import base64
import random

def get_captcha():
    r = int(random.random()*100000000)
    params = {
        'r': str(r),
        's': '0',
    }
    response = requests.get('https://login.sina.com.cn/cgi/pin.php',params=params)
    if response.status_code ==200:
        return response.content


if __name__ == "__main__":
    # 重新下载一张用于测试的验证码
    content = get_captcha()
    with open('current_captcha.jpg','wb') as f:
        f.write(content)
        f.close()

    payload = {
        # 图片的二进制用base64编码一下
        'img':base64.b64encode(content) 
    }
    response = requests.post('http://127.0.0.1:5000/sina',data=payload)
    # print(response.text)
    print('当前目录current_captcha.jpg的识别结果',response.json()['result'])

