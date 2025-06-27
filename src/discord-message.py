import discord
import asyncio
import os 


TOKEN = os.environ['discord_token_fire_alert'] 
CHANNEL_ID = int(os.environ['discord_channel_id_fire_alert'])  # Replace with your channel ID


###############################
def send_message_to_discord(message):

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'Logged in as {client.user}')
        channel = client.get_channel(CHANNEL_ID)
        await channel.send(message)
        await client.close()

    client.run(TOKEN)


if __name__ == '__main__':

    #test
    send_message_to_discord('I am tartampion, and i am going to kick your ass')
