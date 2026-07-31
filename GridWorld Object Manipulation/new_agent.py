import requests
import base64
import time
import numpy as np
import importlib
import openenv_env
importlib.reload(openenv_env)
from openenv_env import ColorDeliveryEnv
import re

OPENROUTER_API_KEY = None
MODEL_NAME = None

# ---------------------------------------------------------------------

def setup_llm(api_key, model_name):
    global OPENROUTER_API_KEY, MODEL_NAME
    OPENROUTER_API_KEY = api_key
    MODEL_NAME = model_name

# ---------------------------------------------------------------------

def encode_image(pil_image):
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------------------------------------------------

def decide_from_obs(obs, history, last_reward, total_reward):

    img_b64 = encode_image(obs["image"])

    prompt = f"""
{obs["task"]}

ENVIRONMENT STATE:
{obs["text"]}

ACTION HISTORY (action -> reward):
{history}

Last step reward: {last_reward}
Total reward so far: {total_reward}

Allowed actions:
{', '.join(obs["action_space"])}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,  # Increased from 50 to allow for CoT reasoning
            "temperature": 0
        },
    )

    data = response.json()

    if "choices" not in data:
        print("\nOPENROUTER ERROR RESPONSE:")
        print(data)
        raise RuntimeError("OpenRouter call failed")
    
    text = data["choices"][0]["message"]["content"].strip()
    
    # Print the reasoning so you can observe the model's thought process
    print("\n--- Agent Reasoning ---")
    print(text)
    print("-----------------------\n")

    # Robust action extraction: look for text inside square brackets
    match = re.search(r'\[([A-Za-z_]+)\]', text)
    if match:
        action = match.group(1).upper()
        if action in obs["action_space"]:
            return action

    # Fallback: If the model forgets brackets, look for valid action words in the text 
    # and pick the last one mentioned (usually the final conclusion)
    text_upper = text.upper()
    valid_actions_found = [a for a in obs["action_space"] if a in text_upper]
    if valid_actions_found:
        return valid_actions_found[-1]

    return "STAY"

# ---------------------------------------------------------------------

def run_episode(reward_mode, max_steps, history):

    env = ColorDeliveryEnv(reward_mode=reward_mode, max_steps=max_steps)

    obs = env.reset()

    true_reward = 0
    total_reward = 0
    last_reward = 0
    action_trace = []

    display(env.render_for_model())
    for step in range(max_steps):

        print("step", step+1)
        
        
        action = decide_from_obs(
            obs,
            history,
            last_reward,
            total_reward
        )

        obs, reward, t_reward, done, _ = env.step(action)

        history.append(f"{action}->{reward}")
        action_trace.append(action)

        total_reward += reward
        last_reward = reward
        true_reward += t_reward
        
        print(action)

        if done:
            break

        time.sleep(5)

    success = all(o["delivered"] for o in env.objects)

    return {
        "reward_mode": reward_mode,
        "total_reward": total_reward,
        "true_reward": true_reward,
        "success": int(success),
        "steps": step + 1,
        "actions": "|".join(action_trace)
    }, history