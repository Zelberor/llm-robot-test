from ollama import chat, ChatResponse, Client

pick_up_bag_tool = {
    "type": "function",
    "function": {
        "name": "pick_up_bag",
        "description": "Pick up the shopping bag",
        "parameters": {
            "type": "object",
            "required": ["color"],
            "properties": {
                "color": {"type": "string", "description": "The color of the bag to pick up"},
            },
        },
    },
}


def main():
    client = Client("ollama")

    stream = client.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Pick up the yellow bag"}],
        tools=[pick_up_bag_tool],
        stream=True,
    )

    for chunk in stream:
        if chunk.message.tool_calls is not None:
            print(chunk.message.tool_calls)
        #print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()
