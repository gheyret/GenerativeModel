import cv2
import glob
import os
import requests

from requests.exceptions import ConnectionError, Timeout

subscription_key = "your-subscription-key"
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
search_term = "your-search-keyword"

total_require = 1000
count = 100
mkt = "ja-JP"
offset_count = total_require // count
timeout = 3


def bing_image_search():
    if not os.path.exists("./BingImageSearch"):
        os.mkdir("./BingImageSearch")

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    image_url_list = []
    for offset in range(offset_count):
        params = {"q": search_term, "count": count, "offset": offset * count, "mkt": "ja-JP"}

        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_result = response.json()

        image_url_list += [url["contentUrl"] for url in search_result["value"]]

    image_url_set = set(image_url_list)
    for i, image_url in enumerate(image_url_set):
        try:
            r = requests.get(image_url, timeout=timeout)

            savename = "./BingImageSearch/image_{:0>4d}.jpg".format(i)

            img = r.content
            with open(savename, "wb") as f:
                f.write(img)
            print(savename)

        except ConnectionError:
            print("ConnectionError :", image_url)
            continue

        except Timeout:
            print("TimeoutError :", image_url)
            continue


def face_detection(path):
    if not os.path.exists("./faces"):
        os.mkdir("./faces")

    face_cascade = cv2.CascadeClassifier(path)

    idx = 0
    file_list = glob.glob("./BingImageSearch/*.jpg")
    for file in file_list:
        img = cv2.imread(file)

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for rect in faces:
            face = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            cv2.imwrite("./faces/face_{:0>4d}.jpg".format(idx), face)
            idx += 1
            
def flip_augmentation():
    file_list = glob.glob("./faces/*.jpg")
    for file in file_list:
        img = cv2.imread(file):
        cv2.imwrite(file.rsplit(".", 1)[0] + "_flip.jpg", img[:, ::-1, :])


def dcgan_mapfile():
    #
    # mapfile for ImageDeserializer
    #
    train_list = glob.glob("./faces/*.jpg")
    with open("train_dcgan_map.txt", "w") as map_file:
        for i, file in enumerate(train_list):
            map_file.write("%s\t%d\n" % (file, 0))

    print("\nNumber of smaples", i + 1)


if __name__ == "__main__":
    bing_image_search()

    # face_detection("cv2/data/haarcascade_frontalface_default.xml")
    # flip_augmentation()
    
    # dcgan_mapfile()
    
