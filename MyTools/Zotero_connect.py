

from pyzotero import zotero
import io
from MyTools.DataIO import secrets

# 1. Connection settings
# Replace with your Zotero User ID and API key.
# For local access, you don't need a public API key, but a private one is recommended for write permissions.
# The `local=True` parameter is crucial for connecting to your local Zotero instance.

zs = secrets()["zotero"]

USER_ID = zs["USER_ID"]
API_KEY = zs["API_KEY"]

zot = zotero.Zotero(USER_ID, 'user', API_KEY, local=True)
zot






# 2. Function to get items by query
def get_zotero_items(query_string, limit=10):
    """Retrieves Zotero items matching a quick search query."""
    print(f"Searching for items with query: '{query_string}'...")
    # The `q` parameter performs a quick search
    # and `qmode` specifies what to search against.
    try:
        items = zot.items(q=query_string, qmode='everything', limit=limit)
        return items
    except Exception as e:
        print(f"An error occurred while searching: {e}")
        return []

# 3. Function to process an item's fields
def process_item_fields(item):
    """Processes an item's metadata and returns a string."""
    title = item['data'].get('title', 'No Title')
    creators = item['data'].get('creators', [])
    author_list = [f"{c['firstName']} {c['lastName']}" for c in creators if c['creatorType'] == 'author']
    authors = ", ".join(author_list) if author_list else "Unknown Author"
    
    # A simple processing example: formatting a summary string
    processed_text = (
        f"Zotero Item Summary\n"
        f"-------------------\n"
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Processed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    return processed_text.encode('utf-8')

# 4. Function to add processed data as an attachment
def add_attachment_to_item(item_key, processed_data):
    """Adds a byte stream as a new attachment file to a Zotero item."""
    print(f"Adding attachment to item with key: {item_key}...")
    try:
        # Use an in-memory buffer to simulate a file
        buf = io.BytesIO(processed_data)
        file_name = f"summary_{item_key}.txt"

        # `zot.attachment_both` uploads the file and creates a new item with the file as an attachment
        # The `parentItem` parameter links it to the original item
        result = zot.attachment_both(
            parentItem=item_key,
            item_data=zot.item_template('attachment'),
            file=buf,
            filename=file_name
        )
        print("Attachment added successfully.")
        return result
    except Exception as e:
        print(f"An error occurred while adding the attachment: {e}")
        return None
    


# if __name__ == "__main__":
#     # Define a user query
#     user_query = "python"
    
#     # Step 1 & 2: Get items
#     my_items = get_zotero_items(user_query)

#     if not my_items:
#         print("No items found. Exiting.")
#     else:
#         print(my_items)
        # for item in my_items:
        #     item_key = item['data']['key']
        #     print(f"\nProcessing item with key: {item_key}")
            
        #     # Step 3: Process fields
        #     processed_content = process_item_fields(item)
            
        #     # Step 4 & 5: Add attachment
        #     add_attachment_to_item(item_key, processed_content)