from claude_api import Client

cookie = 'intercom-device-id-lupk8zyo=e643c36b-8a8c-422c-9aeb-513b13a37f2a; __stripe_mid=306df4a6-d691-4c07-adee-91f4c86819c0fa971a; __ssid=d11f80bf087cff21d9d10e18595c81b; sessionKey=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; activitySessionId=df9f03c0-630b-46c5-a8ef-362422ff8bdd; __cf_bm=.ysxtr0jUUxAI_jQziVBHm9RKUAu0Hnl6wXpuh9wb_0-1696750552-0-AQ2VbiorwXEICb1JHTzEbTx8l2Zrqx+ifbTm5YIUcY0pfaslogpGf/6Gy6ed0Q9yh9wn/Ncemj00bfUxR4nOVqk=; cf_clearance=GTca.pdkgoD6sKqMmxFmTAqAXfbazTlb7b0zFJ.vUaw-1696750553-0-1-1c411939.20e0406c.4b4046d9-0.2.1696750553; intercom-session-lupk8zyo=WmZaWE0yNFBIcWxCWHNJVHp1ekRpTnZNN2VqN0VQanFBd2I0b3dPWno4Q2JHV2h5bkV5MzJTS0w0aGZLbzRqaS0tVzN1WFZXZUQwWXp5cmUxTkk3RU9BQT09--24add038aabac1c42ac32e21360b91fadd4a3e4e'

class Claude():
    def __init__(self, cookie=cookie):
        self.bot = Client(cookie)
        self.conversation_id = "f4bfb242-f74f-4ab2-9482-c5b863f0a89c"
    
    def talk_once(self, prompt="你好"):
        response = self.bot.send_message(prompt, self.conversation_id)
        return response
    
    def list_all_conversations(self):
        conversations = self.bot.list_all_conversations()
        all_id = []
        for conversation in conversations:
            conversation_id = conversation['uuid']
            print(conversation_id)
            all_id.append(conversation_id)
        return all_id
    
    def create_new_conversation(self):
        new_chat = self.bot.create_new_chat()
        conversation_id = new_chat['uuid']
        print("Conversation created successfully, id = ",conversation_id)
        return conversation_id
    
    def delete_conversation(self, conversation_id: str):
        deleted = self.bot.delete_conversation(conversation_id)
        if deleted:
            print("Conversation deleted successfully")
        else:
            print("Failed to delete conversation")


# claude_api = Client(cookie)
# # conversations = claude_api.list_all_conversations()
# for conversation in conversations:
#     conversation_id = conversation['uuid']
#     print(conversation_id)

# prompt.txt = "你好"
# conversation_id = "48b3099d-8b29-4ece-9f1f-da4268dd52bf"
# for i in range(40):
#     response = claude_api.send_message(prompt.txt, conversation_id)
#     print(response)