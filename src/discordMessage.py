import discord
import asyncio
import os 


TOKEN = os.environ['discord_token_fire_alert'] 

###############################
def send_message_to_discord(message, CHANNEL_ID):

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
    CHANNEL_ID = int(os.environ['discord_channel_id_fire_alert_viirs'])  # Replace with your channel ID
    send_message_to_discord('I am tartampion, and i am going to kick your ass', CHANNEL_ID)
    CHANNEL_ID = int(os.environ['discord_channel_id_fire_alert_fci'])  # Replace with your channel ID
    send_message_to_discord('I am tartampion, and i am going to kick your ass', CHANNEL_ID)
