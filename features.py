from bs4 import BeautifulSoup


with open("url_dataset/0.html", encoding='utf-8-sig') as f:
    test = f.read()
soup = BeautifulSoup(test, "html.parser")


def has_title(soup):
    if soup.title is None:
        return 0
    if len(soup.title.text) > 0:
        return 1
    else:
        return 0


def has_input(soup):
    if len(soup.find_all("input")):
        return 1
    else:
        return 0


def has_button(soup):
    if len(soup.find_all("button")) > 0:
        return 1
    else:
        return 0


def has_image(soup):
    if len(soup.find_all("image")) == 0:
        return 0
    else:
        return 1


def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
        else:
            pass
    return 0


def has_link(soup):
    if len(soup.find_all("link")) > 0:
        return 1
    else:
        return 0


def has_password(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("name") or input.get("id")) == "password":
            return 1
        else:
            pass
    return 0


def has_email_input(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("id") or input.get("name")) == "email":
            return 1
        else:
            pass
    return 0


def has_hidden_element(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "hidden":
            return 1
        else:
            pass
    return 0


def has_audio(soup):
    if len(soup.find_all("audio")) > 0:
        return 1
    else:
        return 0


def has_video(soup):
    if len(soup.find_all("video")) > 0:
        return 1
    else:
        return 0


def number_of_inputs(soup):
    return len(soup.find_all("input"))


def number_of_buttons(soup):
    return len(soup.find_all("button"))


def number_of_images(soup):
    image_tags = len(soup.find_all("image"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") or meta.get("name") == "image":
            count += 1
    return image_tags + count


def number_of_option(soup):
    return len(soup.find_all("option"))


def number_of_list(soup):
    return len(soup.find_all("li"))


def number_of_TH(soup):
    return len(soup.find_all("th"))


def number_of_TR(soup):
    return len(soup.find_all("tr"))


def number_of_href(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count


def number_of_paragraph(soup):
    return len(soup.find_all("p"))


def number_of_script(soup):
    return len(soup.find_all("script"))


def length_of_title(soup):
    if soup.title == None:
        return 0
    return len(soup.title.text)



print("has_title --> ", has_title(soup))
print("has_input --> ", has_input(soup))
print("has_button --> ", has_button(soup))
print("has_image --> ", has_image(soup))
print("has_submit --> ", has_submit(soup))
print("has_link --> ", has_link(soup))
print("has_password --> ", has_password(soup))
print("has_email_input --> ", has_email_input(soup))
print("has_hidden_element --> ", has_hidden_element(soup))
print("has_audio --> ", has_audio(soup))
print("has_video --> ", has_video(soup))
print("number_of_inputs --> ", number_of_inputs(soup))
print("number_of_buttons --> ", number_of_buttons(soup))
print("number_of_images --> ", number_of_images(soup))
print("number_of_option --> ", number_of_option(soup))
print("number_of_list --> ", number_of_list(soup))
print("number_of_TH --> ", number_of_TH(soup))
print("number_of_TR --> ", number_of_TR(soup))
print("number_of_href --> ", number_of_href(soup))
print("number_of_paragraph --> ", number_of_paragraph(soup))
print("number_of_script --> ", number_of_script(soup))
print("length_of_title --> ", length_of_title(soup))


def has_h1(soup):
    if len(soup.find_all("h1")) > 0:
        return 1
    else:
        return 0


def has_h2(soup):
    if len(soup.find_all("h2")) > 0:
        return 1
    else:
        return 0


def has_h3(soup):
    if len(soup.find_all("h3")) > 0:
        return 1
    else:
        return 0


def length_of_text(soup):
    return len(soup.get_text())


def number_of_clickable_button(soup):
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count


def number_of_a(soup):
    return len(soup.find_all("a"))


def number_of_img(soup):
    return len(soup.find_all("img"))


def number_of_div(soup):
    return len(soup.find_all("div"))


def number_of_figure(soup):
    return len(soup.find_all("figure"))


def has_footer(soup):
    if len(soup.find_all("footer")) > 0:
        return 1
    else:
        return 0


def has_form(soup):
    if len(soup.find_all("form")) > 0:
        return 1
    else:
        return 0


def has_text_area(soup):
    if len(soup.find_all("textarea")) > 0:
        return 1
    else:
        return 0


def has_iframe(soup):
    if len(soup.find_all("iframe")) > 0:
        return 1
    else:
        return 0


def has_text_input(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "text":
            return 1
    return 0


def number_of_meta(soup):
    return len(soup.find_all("meta"))


def has_nav(soup):
    if len(soup.find_all("nav")) > 0:
        return 1
    else:
        return 0


def has_object(soup):
    if len(soup.find_all("object")) > 0:
        return 1
    else:
        return 0


def has_picture(soup):
    if len(soup.find_all("picture")) > 0:
        return 1
    else:
        return 0


def number_of_sources(soup):
    return len(soup.find_all("source"))


def number_of_span(soup):
    return len(soup.find_all("span"))


def number_of_table(soup):
    return len(soup.find_all("table"))
