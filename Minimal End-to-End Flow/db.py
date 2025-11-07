data_store = []

def save_item(item: str):
    data_store.append(item)
    return True

def get_items():
    return data_store
