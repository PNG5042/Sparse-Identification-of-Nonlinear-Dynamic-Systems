from service import add_item, list_items

def test_add_item():
    # Clear the data_store first to avoid conflicts
    from db import data_store
    data_store.clear()

    response = add_item("test_item")
    assert response == "Item 'test_item' saved successfully!"
    assert "test_item" in list_items()
