import io
import asyncio
import aiohttp
import pandas as pd

async def get_csv_async(client, url):
    async with client.get(url) as response:
        with io.StringIO(await response.text()) as text_io:
            return pd.read_csv(text_io)
        
async def get_json_async(client, url):
    async with client.get(url) as response:
        with io.StringIO(await response.text()) as text_io:
            return pd.read_json(text_io)
        
async def get_excel_async(client, url):
    async with client.get(url) as response:
        with io.StringIO(await response.text()) as text_io:
            return pd.read_excel(text_io)
        
async def get_pd_async(url):
    async with aiohttp.ClientSession() as client:
        future = None
        if url.endswith('.csv'):
            future = get_csv_async(client, url)
        elif url.endswith('.json'):
            future = get_json_async(client, url)
        elif url.endswith('.xlsx'):
            future = get_excel_async(client, url)
        else:
            pass
        return await asyncio.gather(future)
        
