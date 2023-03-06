def prepare_images_txt(path_input, path_output):
    path_input = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib" \
                 "/sfm_compare_pp_calib_4/sparse/model_less_feat/0/images.txt"
    path_output = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib" \
                  "/sfm_compare_pp_calib_4/sparse/model_less_feat/0/images_raw.txt"

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


def pose_permute(path_input, path_output):
    """
    :param path_input: images.txt path
    :param path_output: permuted
    :return:
    """
    path_input = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib" \
                 "/sfm_compare_pp_calib_4/sparse/model/images.txt"
    path_output = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib" \
                  "/sfm_compare_pp_calib_4/sparse/model/images_permuted.txt"

    def add_noise(x, boundry=0.1):
        import random
        var = random.uniform(-boundry, boundry)
        x += x * var
        return x

    with open(path_input) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line != "\n":
                cols = line.split(" ")
                print("old", cols)
                for col in (
                        # 1, 2, 3, 4,
                        5, 6, 7
                ):
                    cols[col] = str(add_noise(float(cols[col]), 0.1))
                lines[index] = " ".join(cols)
                print("new", cols)

        with open(path_output, "w") as fw:
            fw.writelines(lines)
