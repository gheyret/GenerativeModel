import glob


if __name__ == "__main__":
    input_list = glob.glob("./CMP_facade_DB_base/base/*.png")
    with open("./train_pix2pix_x_map.txt", "w") as map_file:
        for i, file in enumerate(input_list):
            map_file.write("%s\t%d\n" % (file, 0))

    truth_list = glob.glob("./CMP_facade_DB_base/base/*.jpg")
    with open("./train_pix2pix_y_map.txt", "w") as map_file:
        for i, file in enumerate(truth_list):
            map_file.write("%s\t%d\n" % (file, 0))
    
