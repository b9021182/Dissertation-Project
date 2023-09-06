import requests as re
import os

folder = "url_dataset"  # Write your own folder name
if not os.path.exists(folder):
    os.mkdir(folder)
def scrape_content(URL):
    response = re.get(URL)
    if response.status_code == 200:
        print("Connected to URL", URL)
        return response
    else:
        print("Could not Connect to URL:", URL)
        return None
path = os.getcwd() + "/" + folder
def save_html(to_where, text, name):
    file_name = name + ".html"
    with open(os.path.join(to_where, file_name), "w", encoding="utf-8") as f:
        f.write(text)
URL_list = [
    "https://www.bbc.com",
    "https://www.youtube.com.",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.instagram.com",
]
def create_url_dataset(to_where, URL_list):
    for i in range(0, len(URL_list)):
        content = scrape_content(URL_list[i])
        if content is not None:
            save_html(to_where, content.text, str(i))
        else:
            pass
    print("URL dataset is created!")
create_url_dataset(path, URL_list)