import os
import certifi
import ssl
import aiohttp
import logging
import discord
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable discord.py logs
for logger_name in ['discord', 'discord.client', 'discord.gateway', 'discord.http']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.handlers.clear()

TOKEN = os.environ['discord_token_rerun']

def send_message_to_discord(message, CHANNEL_ID):


    ssl_context = ssl.create_default_context(cafile=certifi.where())
    intents = discord.Intents.default()

    class MyClient(discord.Client):
        async def setup_hook(self):
            # Patch the aiohttp connector inside the bot’s own event loop
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.http.connector = connector

        async def on_ready(self):
            try:
                channel = self.get_channel(CHANNEL_ID)
                if channel:
                    await channel.send(message)
            finally:
                await self.close()

    client = MyClient(intents=intents)
    client.run(TOKEN)



def send_message_to_discord_viaAeris(message, channel):

    url = f'https://api.sedoo.fr/aeris-euburn-silex-rest/discord/sendMessage/{channel}'
    headers = {
    'accept': '*/*',
    'Content-Type': 'application/json',
    }
    data = message
    response = requests.post(url, headers=headers, data=data, verify=False)
    #print(response.text)
    return response.status_code


if __name__ == '__main__':
    #channel= 'silex-fire-alert-fci'   #int(os.environ['discord_channel_id_fire_alert_fci'])
    #send_message_to_discord_viaAeris('Message for FCI', channel)
    #channel= 'silex-fire-alert-viirs' #int(os.environ['discord_channel_id_fire_alert_viirs'])
    #send_message_to_discord_viaAeris('Message for VIIRS', channel)
    CHANNEL_ID = int(1431283506389975224)
    send_message_to_discord("Test message from Python!", CHANNEL_ID )


