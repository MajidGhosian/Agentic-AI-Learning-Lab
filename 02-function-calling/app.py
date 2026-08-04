import json,os
from dotenv import load_dotenv
from openai import OpenAI
from tool_registry import TOOLS

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tool_schemas=[t["schema"] for t in TOOLS.values()]
messages=[{"role":"system","content":"You are a helpful AI assistant. Use tools whenever they help answer the user."}]

print("Function Calling Demo (type 'exit' to quit)")
while True:
    user=input("\nYou: ")
    if user.lower()=="exit": break
    messages.append({"role":"user","content":user})
    r=client.chat.completions.create(model="gpt-4.1",messages=messages,tools=tool_schemas,tool_choice="auto")
    m=r.choices[0].message
    if m.tool_calls:
        messages.append(m)
        for tc in m.tool_calls:
            fn=TOOLS[tc.function.name]["function"]
            args=json.loads(tc.function.arguments)
            result=fn(**args)
            messages.append({"role":"tool","tool_call_id":tc.id,"content":json.dumps(result)})
        fr=client.chat.completions.create(model="gpt-4.1",messages=messages)
        ans=fr.choices[0].message.content
    else:
        ans=m.content
    print("\nAssistant:",ans)
    messages.append({"role":"assistant","content":ans})
