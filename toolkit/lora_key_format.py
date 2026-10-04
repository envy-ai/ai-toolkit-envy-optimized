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
    if load_key.endswith("$$diff"):
        load_key = load_key[:-len("$$diff")] + ".diff"
    elif load_key.endswith("$$diff_b"):
        load_key = load_key[:-len("$$diff_b")] + ".diff_b"

    if network_type.lower() == "lokr":
        load_key = load_key.replace("$$lokr_w1", ".lokr_w1")
        load_key = load_key.replace("$$lokr_w2", ".lokr_w2")
    if network_type.lower() == "loha":
        for suffix in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"):
            load_key = load_key.replace("$$" + suffix, "." + suffix)
    if load_key.endswith("$$alpha"):
        load_key = load_key[:-7] + ".alpha"

    return load_key
