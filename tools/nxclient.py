import asyncio
import aiohttp
import traceback
from typing import List
from addict import Dict
import os
import cv2
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


NX_URL = os.getenv('NX_URL', 'http://10.233.100.187:5000/v4/NxRead/{model_name}')

def image_to_byte(np_image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, model_image_data = cv2.imencode('.jpg', np_image, encode_param)
    return model_image_data.tobytes()

class NxResultStatus(Enum):
    SUCCESS = 'success'
    ERROR = 'error'


class NxInput(Dict):
    api_key: str
    model_name: str
    image: np.array
    stat_info: str = ''


class NxResult(Dict):
    text: str
    status: NxResultStatus
    message: str
    likelihood: str


class AbsNxClient(ABC):
    @classmethod
    @abstractmethod
    def post_to_nx(cls, nx_inputs: List[NxInput]) -> List[NxResult]:
        pass


class Nx1Result(Dict):
    Status: str
    Message: str
    Text: str
    Rank: str
    Detail: dict
    LoadingParameterDuration: str
    LoadImageDuration: str
    PredictDuration: str
    Likelihood: str


class NxClient(AbsNxClient):
    @staticmethod
    async def async_post(nx_input: NxInput, idx=0):
        # data = {'apiKey': nx_input.api_key, 'image': image_to_byte(nx_input.image), 'statInfo': nx_input.stat_info, 'execFlag': 'true'}
        data = {'apiKey': nx_input.api_key, 'image': image_to_byte(nx_input.image), 'execFlag': 'true'}

        async with aiohttp.ClientSession() as session:
            # create post request
            try:
                url = NX_URL.format(model_name=nx_input.model_name)
                async with session.post(url, data=data, timeout=60000) as response:
                    # wait for response
                    try:
                        js = await response.json(content_type=None)
                        if js['Status'] == 'success':
                            return idx, Nx1Result(js)
                        else:
                            return idx, 'ERROR:{}'.format(js['Message'])
                    except:
                        return idx, 'ERROR:' + traceback.format_exc().replace('\n', '')
            except asyncio.TimeoutError:
                return idx, 'ERROR:asyncio.TimeoutError'
            except:
                return idx, 'ERROR:' + traceback.format_exc().replace('\n', '')

    @classmethod
    def post_to_nx(cls, nx_inputs: List[NxInput]) -> List[NxResult]:
        if len(nx_inputs) == 0:
            return []
        loop = asyncio.new_event_loop()
        tasks = [cls.async_post(nx_i, i) for i, nx_i in enumerate(nx_inputs)]
        api_res = loop.run_until_complete(asyncio.wait(tasks))
        api_res = [r.result() for r in api_res[0]]

        ocr_results = []
        for r in sorted(api_res):
            r = r[1]
            if type(r) is Nx1Result:
                ocr_results.append(NxResult(text=r.Text, status=NxResultStatus.SUCCESS, message=r.Message, likelihood=r.Likelihood))
            else:
                message = r if r is str else ''
                ocr_results.append(NxResult(text='', status=NxResultStatus.ERROR, message=message, likelihood=0))

        return ocr_results
