import sys


def prepare_images_txt(path_input, path_output):
    with open(path_input) as f:
        lines = f.readlines()

    images_txt_new = []

    for i in range(4, len(lines)):
        if i % 2 == 0:
            images_txt_new.append(lines[i])
        else:
            images_txt_new.append("")

    with open(path_output, "w") as outfile:
        outfile.write("\n".join(images_txt_new))


prepare_images_txt(sys.argv[1], sys.argv[2])
