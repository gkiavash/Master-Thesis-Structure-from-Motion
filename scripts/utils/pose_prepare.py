PATH_MODEL = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib_3_prato/sfm_compare_pp_calib/sparse/0/export_as_txt/images.txt"
PATH_OUTPUT = "/home/gkiavash/Downloads/sfm_projects/sfm_compare_4_pp_calib_3_prato/sfm_compare_pp_calib/sparse/0/export_as_txt/images_raw.txt"


def prepare_images_txt(path_model, path_output):
    with open(path_model) as f:
        lines = f.readlines()

    images_txt_new = []

    for i in range(4, len(lines)):
        if i % 2 == 0:
            images_txt_new.append(lines[i])
        else:
            images_txt_new.append("")

    with open(path_output, "w") as outfile:
        outfile.write("\n".join(images_txt_new))


prepare_images_txt(PATH_MODEL, PATH_OUTPUT)
