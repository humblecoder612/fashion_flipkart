import requests
import bs4
import io
import shutil

def site_scraper(url):
    url=url
    response=requests.get(url)
    filename="temp.html"
    bs=bs4.BeautifulSoup(response.text,"html.parser")
    formatted_text=bs.prettify()
    # print(formatted_text)
    try:
        with io.open(filename, "w+", encoding="utf-8") as f:
            f.write(formatted_text)
    except Exception  as e:
            print(e)


    img_list=bs.find_all('img')

    j=1
    for imgTag in img_list:
        try:
            imgLink=imgTag.get('src')
            print(imgLink)
            ext=imgLink[imgLink.rindex('.'):]
            if ext.startswith(".png"):
                ext=".png"
            elif ext.startswith(".jpeg"):
                ext=".jpeg"
            elif ext.startswith("jpg"):
                ext=".jpg"
            elif ext.startswith(".svg"):
                ext=".svg"            
            filen=str(j)+ext
            res=requests.get(imgLink,stream=True)
            with io.open('scraped/images/%s'%(filen), "wb") as files:
                shutil.copyfileobj(res.raw,files)
            try:
                with io.open('scraped/urls/%s'%(j), "w+", encoding="utf-8") as f:
                    f.write(imgLink)
            except Exception  as ee:
                 print(ee)
        except Exception  as e:
            print(e)

    j=j+1


