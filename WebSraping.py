import re
import json
import requests
import pandas as pd

#exit()
url = "https://shopee.co.id/NIVEA-Body-Lotion-Daily-Protection-Sun-Lotion-SPF33-PA-100ml-i.39283823.603269145?xptdk=abe276e2-fbce-435f-9585-9eda6f617950"
r = re.search(r"i\.(\d+)\.(\d+)", url)
shop_id, item_id = r[1], r[2]
ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"
offset = 0
d = {"username": [], "rating": [], "comment": []}

while True:
    data = requests.get(
        ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
    ).json()

    # uncomment this to print all data:
    # print(json.dumps(data, indent=4))
    
    i = 1
    
    if data["data"]["ratings"] is not None:
        try:
            for i, rating in enumerate(data["data"]["ratings"], 1):
                    d["username"].append(rating["author_username"])
                    d["rating"].append(rating["rating_star"])
                    d["comment"].append(rating["comment"])

                    print(rating["author_username"])
                    print(rating["rating_star"])
                    print(rating["comment"])
                    print("-" * 100)
        except:
            print("Error")
            break


    if i % 20:
        break

    offset += 20
   

df = pd.DataFrame(d)
print(df)
df.to_csv("data.csv", index=False)