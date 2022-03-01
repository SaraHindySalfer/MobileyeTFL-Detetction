from part4.controller import Controller


def create_play_list(clip_name, folder_path, frames_number):
    with open(clip_name + ".pls", 'w') as file_writer:
        file_writer.write(folder_path + "/" + clip_name + ".pkl\n")
        for i in range(frames_number):
            file_writer.write(folder_path + "/" + clip_name + "_0000" + str(i+24) + "_leftImg8bit.png\n")


def main():
    # create_play_list("dusseldorf_000049", "C:/Users/RENT/Desktop/dissuldorf_49",6)
    control = Controller("./dusseldorf_000049.pls")
    control.run()


if __name__ == '__main__':
    main()
