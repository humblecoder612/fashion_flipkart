import megapints

keywords=[
"80s fashion t-shirts",
"fashion designing t-shirts",
"fashion designer t-shirts",
"fashion blog t-shirts",
"mens fashion t-shirts",
"fashion designer games t-shirts",
"fashion show t-shirts",
"fashion week t-shirts",
"fashion tv t-shirts",
"new york fashion week t-shirts",
"fashion dresses t-shirts",
"fashion model t-shirts",
"fashion style t-shirts",
"fashion magazine t-shirts",
"korean fashion t-shirts",
"men fashion t-shirts",
"fashion 2016 t-shirts",
"new fashion t-shirts",
"london fashion week t-shirts",
"fashion illustration t-shirts",
"fashion trends t-shirts",
"women fashion t-shirts",
"fashion accessory t-shirts"]

sites=["Blog.patternbank.com","https://www.collezioni.info/en/tag/menswear/","https://www.vogue.com/tag/misc/menswear",'https://www.google.com/search?q=men+fashion+trend+tshirts&tbm=isch&ved=2ahUKEwj5n9yR7IzrAhXU9jgGHaTWDbUQ2-cCegQIABAA&oq=men+fashion+trend+tshirts&gs_lcp=CgNpbWcQAzoECCMQJzoCCAA6BAgAEB46BggAEAUQHjoGCAAQCBAeUNdOWP5kYJVmaAFwAHgAgAF-iAGOCJIBAzAuOZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=hEAvX_nNI9Tt4-EPpK23qAs&bih=921&biw=1920','https://www.apetogentleman.com/mens-fashion-trends/']

def dataset_update():
    for ele in keywords:
        megapints.pin_down(ele,5)
    from datetime import date
    from datetime import date
    week=str(date.today())

    import os
    import siter
    from distutils.dir_util import copy_tree
    os.mkdir(week)
    for i in range(len(sites)):
        os.mkdir('./'+week+'/'+str(i))
        try:
            siter.site_scraper(sites[i])
            copy_tree("./scraped", week+'/'+str(i))
        except:
            continue

