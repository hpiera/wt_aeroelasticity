import os

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = r"figures"
    folder = os.path.join(dir_path,path)
    for count, filename in enumerate(os.listdir(folder)):
        original = filename
        filename = filename[:-4]
        filename = filename.replace(".","_")
        filename = filename + ".pdf"

        src = f"{folder}/{original}"
        dst = f"{folder}/{filename}"
        print(filename)


        # rename() function will
        # rename all the files
        os.rename(src, dst)


if __name__ == '__main__':
    main()
