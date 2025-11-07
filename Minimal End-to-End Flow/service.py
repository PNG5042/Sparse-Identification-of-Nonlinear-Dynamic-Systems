from db import save_item, get_items

def add_item(item: str):
    success = save_item(item)
    if success:
        return f"Item '{item}' saved successfully!"
    return "Failed to save item"

def list_items():
    return get_items()
