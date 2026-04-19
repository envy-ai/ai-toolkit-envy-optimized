def internal_key_to_peft_key(key: str) -> str:
    new_key = key.replace("lora_down", "lora_A")
    new_key = new_key.replace("lora_up", "lora_B")
    return new_key.replace("$$", ".")


def peft_key_to_internal_key(key: str, network_type: str = "lora") -> str:
    load_key = key.replace("lora_A", "lora_down")
    load_key = load_key.replace("lora_B", "lora_up")
    load_key = load_key.replace(".", "$$")

    load_key = load_key.replace("$$lora_down$$", ".lora_down.")
    load_key = load_key.replace("$$lora_up$$", ".lora_up.")
    load_key = load_key.replace("$$magnitude", ".magnitude")

    if network_type.lower() == "lokr":
        load_key = load_key.replace("$$lokr_w1", ".lokr_w1")
        load_key = load_key.replace("$$lokr_w2", ".lokr_w2")
        if load_key.endswith("$$alpha"):
            load_key = load_key[:-7] + ".alpha"

    return load_key
