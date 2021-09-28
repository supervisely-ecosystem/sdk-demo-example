import supervisely_lib as sly

import cv2
import copy

# Init API variables
address = 'https://app.supervise.ly/'

token = ''  # paste from https://app.supervise.ly/user/settings/tokens

# Initialize API object
api = sly.Api(address, token)


"""
STEP 1 — Creating and uploading project
"""


# Initialize annotation(-s)
class_defect = sly.ObjClass('defect', sly.Rectangle)

label_defect = sly.Label(sly.Rectangle(top=46, left=122, bottom=114, right=180), class_defect)

test_image = cv2.imread('./test_image.png')
height, width = test_image.shape[0], test_image.shape[1]

defect_annotation = sly.Annotation((height, width), [label_defect])


# Init project remotely
workspace_id = 60361

project_name = 'demo_project'
dataset_name = 'demo_dataset'

project = api.project.create(workspace_id, project_name, type=sly.ProjectType.IMAGES,
                             change_name_if_conflict=True)
dataset = api.dataset.create(project.id, f'{dataset_name}',
                             change_name_if_conflict=True)


# Updating meta
objects = sly.ObjClassCollection([class_defect])
project_meta = sly.ProjectMeta(obj_classes=objects, project_type='images')

updated_meta = api.project.update_meta(project.id, project_meta.to_json())


# Uploading image
img_info = api.image.upload_path(dataset.id, name="test_image.png", path="./test_image.png")

# Uploading annotation
uploaded_annotation_json = api.annotation.upload_json(img_info.id, defect_annotation.to_json())


"""
STEP 2 — Downloading and visualizing results
"""

# Downloading image
ann_info = api.annotation.download(img_info.id)
ann = sly.Annotation.from_json(ann_info.annotation, project_meta)

# Downloading annotations
img = api.image.download_np(img_info.id)
new_img = copy.deepcopy(img)

# Draw Annotation on image before crop
ann.draw_pretty(img, thickness=1, output_path='./test_image_annotated.png')






