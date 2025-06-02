from ollama import Client
import os

get_status_tool = {
    "type": "function",
    "function": {
        "name": "get_robot_status",
        "description": "Get the current status of the robot",
    },
}

list_bags_tool = {
    "type": "function",
    "function": {
        "name": "list_bags",
        "description": "Get a list of shopping bags the robot currently sees",
    },
}

pick_up_bag_tool = {
    "type": "function",
    "function": {
        "name": "pick_up_bag",
        "description": "Pick up the shopping bag",
        "parameters": {
            "type": "object",
            "required": ["color"],
            "properties": {
                "color": {
                    "type": "string",
                    "description": "The color of the bag to pick up",
                },
            },
        },
    },
}

follow_me_tool = {
    "type": "function",
    "function": {
        "name": "follow_me",
        "description": "Follow the user",
    },
}


def call_model(client: Client, messages: list):
    stream = client.chat(
        model="arl-robot-assistant",
        messages=messages,
        tools=[pick_up_bag_tool, list_bags_tool, follow_me_tool, get_status_tool],
        think=True,
        stream=True,
    )
    return stream


def main():
    host = os.getenv("OLLAMA_HOST", "localhost")
    client = Client(host=host)
    response = client.create(
        model="arl-robot-assistant",
        from_="qwen3:8b",
        system="""
You are Zirbi, a physical home-appliance robot.
You currently take part in the RoboCup@home competition.
Your job is to assist the users and complete tasks.

Always state you name when greeting. Do not offer your help when greeting.

You have several tools at your disposal that let you control your physical hardware.
Only use tools when explicitly asked.
If the user mentions an parameter for a tool, only offer your help but do not execute the tool right away.
Always give feedback about the tool's execution.

When someone refers to a robot, they are referring to you.

You should always answer casually but precisely.
The chat should look like a normal casual conversation.
If you are asked a question, answer it directly.
Do not add unnecessary information like conclusions or explanations unless asked.

Only think for a short time, the user needs your answer quickly.

Your responses are fed into a text to speech model.
""",
        stream=False,
    )
    print(f"Model status: {response.status}")

    messages = []

    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})

        print("Robot: ", end="", flush=True)

        stream = call_model(client, messages)

        tool_calls = []
        robot_answer = ""
        for chunk in stream:
            text_chunk = chunk.message.content
            if text_chunk is not None:
                print(text_chunk, end="", flush=True)
                robot_answer += text_chunk

            if chunk.message.tool_calls is not None:
                tool_calls.extend(chunk.message.tool_calls)
        messages.append({"role": "assistant", "content": robot_answer})

        for tool_call in tool_calls:
            print(f"\n>>LLM executed tool call: {tool_call}<<")
            function_name = tool_call.function.name
            if function_name == "follow_me":
                # TODO: Send system command to robot to follow the user
                tool_response = "Now following the user. The robot now follows the user to their destination"
            elif function_name == "pick_up_bag":
                tool_response = "Starting to pick up bag..."
            elif function_name == "list_bags":
                tool_response = "[green, purple]"
            elif function_name == "get_robot_status":
                tool_response = "Chilling out..."
            else:
                tool_response = "Unknown tool call"
            messages.append(
                {
                    "role": "tool",
                    "content": tool_response,
                    "name": function_name,
                }
            )
        if len(tool_calls) > 0:
            robot_answer = ""
            stream = call_model(client, messages)
            for chunk in stream:
                text_chunk = chunk.message.content
                if text_chunk is not None:
                    print(text_chunk, end="", flush=True)
                    robot_answer += text_chunk
            messages.append({"role": "assistant", "content": robot_answer})

        print("\n", flush=True)


if __name__ == "__main__":
    main()
