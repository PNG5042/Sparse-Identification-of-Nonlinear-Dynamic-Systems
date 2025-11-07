from service import add_item, list_items

def main():
    print("Welcome to the minimal app demo!")
    item = input("Enter item to save: ")
    print(add_item(item))
    print("Current items:", list_items())

if __name__ == "__main__":
    main()
