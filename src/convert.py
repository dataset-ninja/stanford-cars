import csv
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import scipy
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    
    train_path = os.path.join("Stanford Cars","cars_train","cars_train")
    test_path = os.path.join("Stanford Cars","cars_test","cars_test")
    anns_file = os.path.join("Stanford Cars","cars_annos.mat")  # anns not correct, only take classes names

    # get bboxes from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/discussion/281003
    train_anns = os.path.join("Stanford Cars","cardatasettrain.csv")
    test_anns = os.path.join("Stanford Cars","cardatasettest.csv")
    batch_size = 30


    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        im_name = get_file_name_with_ext(image_path)
        ann = name_to_ann.get(im_name)

        for curr_ann in ann:
            obj_class = idx_to_class[curr_ann[0]]

            left = curr_ann[1][0]
            right = curr_ann[1][2]
            top = curr_ann[1][1]
            bottom = curr_ann[1][3]
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    mat = scipy.io.loadmat(anns_file)

    car = sly.ObjClass("car", sly.Rectangle)
    idx_to_class = {0: car}
    classes_names = mat["class_names"][0]
    for idx, class_name_arr in enumerate(classes_names):
        class_name = str(class_name_arr[0]).lower()
        class_name_corr = "_".join(class_name.split())
        obj_class = sly.ObjClass(class_name_corr, sly.Rectangle)
        idx_to_class[idx + 1] = obj_class

    train_name_to_ann = defaultdict(list)
    test_name_to_ann = defaultdict(list)

    with open(train_anns, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            train_name_to_ann[row[6]].append(
                [int(row[5]), [int(row[1]), int(row[2]), int(row[3]), int(row[4])]]
            )

    with open(test_anns, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            test_name_to_ann[row[5]].append([0, [int(row[1]), int(row[2]), int(row[3]), int(row[4])]])

    ds_name_to_data = {"train": (train_path, train_name_to_ann), "test": (test_path, test_name_to_ann)}


    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=list(idx_to_class.values()))
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        data_path, name_to_ann = ds_data
        images_names = os.listdir(data_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(data_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
